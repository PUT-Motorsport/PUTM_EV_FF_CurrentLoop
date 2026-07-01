#!/usr/bin/env python3
"""
CurrentPredictorNode — 1-step-ahead motor current predictor for the 80 kW power limiter.

Subscribes (configure topic names in config/params.yaml or via launch args):
  current_topic  [std_msgs/Float32MultiArray]  I_FL, I_FR, I_RL, I_RR [raw ADC]
                 Actual VCL topic: /putm_vcl/current_sensor (custom msg, see ADAPTER section)
  torque_topic   [std_msgs/Float32MultiArray]  T_FL, T_FR, T_RL, T_RR [Nm]
                 Actual VCL topic: /putm_vcl/setpoints       (custom msg, see ADAPTER section)
  voltage_topic  [std_msgs/Float32]            U_dc [V]
                 Actual VCL topic: /putm_vcl/bms_hv_main     (custom msg, see ADAPTER section)

Publishes at 5 Hz:
  ~/predicted_currents      Float32MultiArray  max(ARX, XGBoost) per motor [I_FL,I_FR,I_RL,I_RR]
  ~/predicted_currents_arx  Float32MultiArray  ARX Q=0.90 predictions
  ~/predicted_currents_xgb  Float32MultiArray  XGBoost Q=0.90 predictions
  ~/predicted_power_kw      Float32            U_dc * sum(predicted_currents) [kW]
  ~/power_limit_exceeded    Bool               True if EITHER model predicts > 80 kW
  ~/diagnostics             DiagnosticArray    per-model details + inference times

ADAPTER SECTION:
  The _cb_currents, _cb_torques, _cb_voltage methods use std_msgs placeholders.
  Replace their bodies when you add the actual putm_vcl message imports.
  The rest of the node does not need to change.
"""

import time
import struct
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from ament_index_python.packages import get_package_share_directory

from .lag_buffer import LagBuffer
from .arx_predictor import ARXPredictor
from .xgb_predictor import XGBPredictor

POWER_LIMIT_W = 80_000.0


class CurrentPredictorNode(Node):

    def __init__(self) -> None:
        super().__init__('current_predictor')

        self.declare_parameter('current_topic',    'current_sensors_data')
        self.declare_parameter('torque_topic',     'torque_setpoints')
        self.declare_parameter('voltage_topic',    'bus_voltage')
        self.declare_parameter('n_lags',           5)
        self.declare_parameter('publish_rate_hz',  5.0)
        self.declare_parameter('power_limit_w',    POWER_LIMIT_W)
        pkg_share = get_package_share_directory('current_predictor')
        self.declare_parameter('weights_dir', str(Path(pkg_share) / 'model_weights'))

        weights_dir  = Path(self.get_parameter('weights_dir').value)
        n_lags       = self.get_parameter('n_lags').value
        self._limit  = self.get_parameter('power_limit_w').value

        self.get_logger().info(f'Loading models from {weights_dir}')
        self._arx    = ARXPredictor(str(weights_dir / 'arx_q90_weights.json'))
        self._xgb    = XGBPredictor(str(weights_dir))
        self.get_logger().info('Models loaded (ARX Q=0.90 + XGBoost Q=0.90)')

        self._buffer = LagBuffer(n_lags=n_lags)

        # Latest cached sensor values (updated by callbacks)
        self._currents: np.ndarray = np.zeros(4, dtype=np.float64)
        self._t_sum: float = 0.0
        self._u_dc: float  = 0.0
        self._has_current_data: bool = False

        cur_topic = self.get_parameter('current_topic').value
        trq_topic = self.get_parameter('torque_topic').value
        vlt_topic = self.get_parameter('voltage_topic').value

        self.create_subscription(Float32MultiArray, cur_topic, self._cb_currents, 10)
        self.create_subscription(Float32MultiArray, trq_topic, self._cb_torques,  10)
        self.create_subscription(Float32,           vlt_topic, self._cb_voltage,  10)

        self._pub_currents     = self.create_publisher(Float32MultiArray, '~/predicted_currents',     10)
        self._pub_currents_arx = self.create_publisher(Float32MultiArray, '~/predicted_currents_arx', 10)
        self._pub_currents_xgb = self.create_publisher(Float32MultiArray, '~/predicted_currents_xgb', 10)
        self._pub_power        = self.create_publisher(Float32,           '~/predicted_power_kw',     10)
        self._pub_flag         = self.create_publisher(Bool,              '~/power_limit_exceeded',   10)
        self._pub_diag         = self.create_publisher(DiagnosticArray,   '~/diagnostics',            10)

        rate = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / rate, self._predict_and_publish)

        self.get_logger().info(
            f'CurrentPredictorNode ready at {rate:.0f} Hz '
            f'(topics: {cur_topic}, {trq_topic}, {vlt_topic})'
        )

    # ----------------------------------------------------------------
    # ADAPTER CALLBACKS
    # Replace message types and field access to match putm_vcl messages.
    # Expected output of each method is documented below.
    # ----------------------------------------------------------------

    def _cb_currents(self, msg: Float32MultiArray) -> None:
        """Parse current sensor data -> [I_FL, I_FR, I_RL, I_RR] (raw ADC units).

        Placeholder: expects Float32MultiArray with 4 elements in motor order.

        To use the actual /putm_vcl/current_sensor topic:
          1. Add the custom message import (e.g. putm_vcl_msgs.msg.CurrentSensor)
          2. Replace the subscription type above
          3. Replace msg.data[:4] with the actual field names
          Note: raw uint16 values, CURRENT_SCALE = 1.0 currently.
        """
        self._currents = np.array(msg.data[:4], dtype=np.float64)
        self._has_current_data = True

    def _cb_torques(self, msg: Float32MultiArray) -> None:
        """Parse torque setpoints -> T_sum = T_FL + T_FR + T_RL + T_RR [Nm].

        Placeholder: expects Float32MultiArray with 4 elements [T_FL, T_FR, T_RL, T_RR].

        To use /putm_vcl/setpoints:
          1. Import the custom message type
          2. Replace msg.data[:4] with actual int32 field access
          Note: TORQUE_SCALE = 1.0 (values already in Nm).
        """
        self._t_sum = float(sum(msg.data[:4]))

    def _cb_voltage(self, msg: Float32) -> None:
        """Parse DC bus voltage -> U_dc [V].

        Placeholder: expects Float32 in Volts.

        To use /putm_vcl/bms_hv_main:
          1. Import the custom message type
          2. Apply VOLTAGE_SCALE: self._u_dc = float(msg.voltage_sum) * 0.1
          Note: raw uint16 / 10 = V (e.g. 5633 raw = 563.3 V).
        """
        self._u_dc = float(msg.data)

    # ----------------------------------------------------------------

    def _predict_and_publish(self) -> None:
        if not self._has_current_data:
            return

        self._buffer.update(self._currents, self._t_sum, self._u_dc)

        if not self._buffer.ready:
            return

        x = self._buffer.get_feature_vector()   # (31,)

        t0 = time.perf_counter()
        i_arx = self._arx.predict(x)            # (4,)
        t_arx = time.perf_counter() - t0

        t0 = time.perf_counter()
        i_xgb = self._xgb.predict(x)            # (4,)
        t_xgb = time.perf_counter() - t0

        # Element-wise maximum: most conservative estimate per motor
        i_final = np.maximum(i_arx, i_xgb)

        p_arx_w   = self._u_dc * i_arx.sum()
        p_xgb_w   = self._u_dc * i_xgb.sum()
        p_final_w = self._u_dc * i_final.sum()
        p_final_kw = p_final_w / 1000.0

        # Flag: either model exceeds the limit
        limit_exceeded = bool(p_arx_w > self._limit or p_xgb_w > self._limit)

        self._pub_currents.publish(_make_f32arr(i_final))
        self._pub_currents_arx.publish(_make_f32arr(i_arx))
        self._pub_currents_xgb.publish(_make_f32arr(i_xgb))

        msg_p = Float32()
        msg_p.data = float(p_final_kw)
        self._pub_power.publish(msg_p)

        msg_flag = Bool()
        msg_flag.data = limit_exceeded
        self._pub_flag.publish(msg_flag)

        self._publish_diagnostics(i_arx, i_xgb, i_final, p_arx_w, p_xgb_w, t_arx, t_xgb)

    def _publish_diagnostics(
        self,
        i_arx: np.ndarray,
        i_xgb: np.ndarray,
        i_final: np.ndarray,
        p_arx_w: float,
        p_xgb_w: float,
        t_arx: float,
        t_xgb: float,
    ) -> None:
        level = DiagnosticStatus.WARN if (p_arx_w > self._limit or p_xgb_w > self._limit) \
                else DiagnosticStatus.OK

        motors = ('FL', 'FR', 'RL', 'RR')
        kv: list[KeyValue] = []
        for i, m in enumerate(motors):
            kv.append(KeyValue(key=f'I_{m}_arx',   value=f'{i_arx[i]:.1f}'))
            kv.append(KeyValue(key=f'I_{m}_xgb',   value=f'{i_xgb[i]:.1f}'))
            kv.append(KeyValue(key=f'I_{m}_final', value=f'{i_final[i]:.1f}'))
        kv += [
            KeyValue(key='power_arx_kw',   value=f'{p_arx_w / 1000:.2f}'),
            KeyValue(key='power_xgb_kw',   value=f'{p_xgb_w / 1000:.2f}'),
            KeyValue(key='U_dc_V',         value=f'{self._u_dc:.1f}'),
            KeyValue(key='t_arx_ms',       value=f'{t_arx * 1e3:.4f}'),
            KeyValue(key='t_xgb_ms',       value=f'{t_xgb * 1e3:.4f}'),
        ]

        status = DiagnosticStatus(
            name='current_predictor',
            hardware_id='VCU',
            level=level,
            message='power limit exceeded' if level == DiagnosticStatus.WARN else 'OK',
            values=kv,
        )
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [status]
        self._pub_diag.publish(arr)


def _make_f32arr(values: np.ndarray) -> Float32MultiArray:
    msg = Float32MultiArray()
    msg.data = values.tolist()
    return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CurrentPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
