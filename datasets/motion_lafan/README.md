# Q1 v3 LaFAN1 动作数据

将已经重定向至 Q1 v3 22 自由度的 LaFAN1 CSV 放在此目录。CSV 被 `.gitignore`
排除，不随 Git 提交发布。

每个 CSV 无表头、30 Hz，每帧 **29** 个浮点数，依次为：

1. 根位置 `x, y, z`，单位米。
2. 根姿态四元数 `qx, qy, qz, qw`。
3. **22** 个关节角，单位弧度，顺序与 `supermimic/robot.py` 的 `JOINT_NAMES` 一致
   （左腿 6、右腿 6、`waist_roll`、`waist_yaw`、左臂 4、右臂 4）。

读取时重采样至 50 Hz，速度通过有限差分计算。这里需要 **Q1 v3** 重定向版本；
E1 的 21 自由度（28 列）或 G1 的 29 自由度 CSV 都不能直接使用——Q1 v3 比 E1
多了 `waist_roll_joint`，列数与关节顺序均不同。

数据来源和许可范围见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
