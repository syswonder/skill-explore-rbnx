# SPDX-License-Identifier: MulanPSL-2.0
# explore_rbnx — frontier-based exploration skill image.
#
# Lightweight ROS Humble base with rclpy + numpy/scipy for grid math.
# No SLAM packages here — this skill consumes mapping's output via
# atlas-resolved /map and drives via the navigation service's gRPC.
# --network host + --ipc=host at runtime so DDS reaches /map and the
# nav service is reachable on 127.0.0.1.
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime deps:
#   tf2 — pose lookup (we may need map→base_link to know where we are
#         relative to a frontier candidate)
#   numpy/scipy — grid + clustering math (frontier + DBSCAN)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-yaml \
        ros-humble-tf2-ros \
        ros-humble-tf-transformations \
 && rm -rf /var/lib/apt/lists/*

# Python deps:
#   grpcio/protobuf — atlas client
#   fastmcp/uvicorn — MCP server for the 3 skill tools (LLM-callable)
RUN pip install --no-cache-dir \
        grpcio>=1.60.0 \
        protobuf>=4.25.0 \
        pyyaml>=6.0 \
        numpy \
        fastmcp \
        uvicorn

WORKDIR /explore
COPY entrypoint.sh /entrypoint.sh
COPY no_shm_profile.xml /etc/fastrtps_no_shm.xml
RUN chmod +x /entrypoint.sh

# Match mapping's RMW choice so cross-container DDS works on the same
# bus without bridge magic (FastRTPS UDP-only profile).
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/etc/fastrtps_no_shm.xml
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ENTRYPOINT ["/entrypoint.sh"]
