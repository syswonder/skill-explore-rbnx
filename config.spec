# The mapping under the Explore skill's `config:` in robonix_manifest.yaml,
# delivered through Driver(CMD_INIT). A bad strategy name fails CMD_INIT.
#
# Per-request limits are not here: area_hint, timeout_s and max_speed_m_s are
# fields of each robonix/skill/explore/explore request. The robot's size is
# not here either: it is read from Soma (robonix/system/soma/footprint), the
# inscribed radius plus 0.05 m.

specVersion: 1
description: >-
  How the Explore skill chooses the next frontier to visit.

properties:
  reference:
    type: object
    description: Inherited deployment reference for robot duration and data age.
    properties:
      kind:
        type: string
        enum: [linux_system, ros_time]
        default: linux_system
      provider_id:
        type: string
        default: chronos

  strategy:
    type: string
    enum: [frontier_greedy, mrtsp]
    default: frontier_greedy
    description: >-
      frontier_greedy goes to the frontier that scores best from here, its
      size over the straight-line distance. mrtsp follows Liu et al. (2025):
      frontiers are searched on a map cleaned of the specks sparse lidar
      beams leave, put in the visiting order of the minimum-ratio travelling
      salesman problem, and the first one is visited.

  mrtsp:
    type: object
    description: >-
      Parameters of the mrtsp strategy, Eq. 5-8 of the paper. Ignored by
      frontier_greedy. The paper gives no values for the weights.
    properties:
      wd:
        type: number
        default: 1.0
        minimum: 0
        description: Weight of the path cost, Wd in Eq. 7.
      ws:
        type: number
        default: 1.0
        minimum: 0
        description: Weight of the information gain, Ws in Eq. 7. Must be greater than zero; CMD_INIT refuses 0.
      sensor_range_m:
        type: number
        default: 0.0
        minimum: 0
        description: >-
          Metres; r_s in Eq. 5, subtracted from the path cost. Above zero, the
          smallest frontier within reach becomes the cheapest move.
      min_frontier_m:
        type: number
        default: 1.0
        minimum: 0
        description: Metres; shorter frontiers are not candidates.
      vmax_m_s:
        type: number
        default: 0.35
        minimum: 0
        description: Metres per second; the robot's top speed, v_max in Eq. 8. Must be greater than zero; CMD_INIT refuses 0.
      wmax_rad_s:
        type: number
        default: 0.6
        minimum: 0
        description: Radians per second; the robot's top turn rate, w_max in Eq. 8. Must be greater than zero; CMD_INIT refuses 0.
