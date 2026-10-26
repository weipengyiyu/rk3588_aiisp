/* SPDX-License-Identifier: ((GPL-2.0+ WITH Linux-syscall-note) OR MIT)
 * Minimal userspace subset of this BSP's kernel/include/uapi/linux/rk-isp2-config.h.
 * Copyright (C) 2017 Rockchip Electronics Co., Ltd.
 */
#pragma once
#include <linux/types.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>

enum isp2x_trigger_mode { T_START_X1=1U<<4 };
struct isp2x_csi_trigger {
    __u64 sof_timestamp;
    __u64 frame_timestamp;
    __u32 frame_id;
    int times;
    enum isp2x_trigger_mode mode;
} __attribute__((packed));
static_assert(sizeof(isp2x_csi_trigger)==28,"BSP trigger ABI mismatch");
constexpr int CSI_MEM_WORD_LITTLE_ALIGN=1;
#define RKISP_CMD_TRIGGER_READ_BACK _IOW('V', BASE_VIDIOC_PRIVATE + 0, struct isp2x_csi_trigger)
#define RKISP_CMD_GET_CSI_MEMORY_MODE _IOR('V', BASE_VIDIOC_PRIVATE + 100, int)
#define RKISP_CMD_SET_CSI_MEMORY_MODE _IOW('V', BASE_VIDIOC_PRIVATE + 101, int)
