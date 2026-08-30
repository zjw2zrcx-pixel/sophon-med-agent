---
name: "room_exploration"
description: "在室内环境中系统地探索房间以寻找目标"
when_to_use: "当需要搜索目标物体但不确定其位置时"
mode: "Voice"
allowed_tools:
  - navigate_to
  - describe_scene
  - search_object
---

# 房间探索技能

1. 使用 describe_scene 观察当前环境
2. 根据目标物体属性判断最可能所在的房间类型
3. 使用 navigate_to 前往最可能的房间
4. 到达后使用 search_object 确认目标
5. 如果未找到，更新判断并前往下一个候选房间
