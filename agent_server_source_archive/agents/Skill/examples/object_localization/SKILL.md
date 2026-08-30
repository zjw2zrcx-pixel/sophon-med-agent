---
name: "object_localization"
description: "在已知房间中定位特定物体"
when_to_use: "当已知目标所在房间但需要找到其精确位置时"
mode: "Voice"
allowed_tools:
  - describe_scene
  - search_object
  - look_at
---

# 物体定位技能

1. 使用 describe_scene 观察当前房间全景
2. 根据目标物体属性(颜色、大小、形状)判断可能位置
3. 使用 look_at 转向可能的方向
4. 使用 search_object 精确搜索
5. 确认找到目标后报告位置
