"""挂接、换钩与解钩序列生成：双钩交替前进，任何状态至少保留一个有效连接。

算法：把路线离散为站点后，对每个站点求可到达锚点集；双钩沿站点推进，
当某钩锚点在下一站将超出触及范围时，必须在当前站把该钩换到新的前向锚点
（换钩期间另一钩保持连接）。若当前站找不到满足触及与共用限制的锚点，
则换挂链断开——继续走将出现双钩同时解开的瞬间。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (Anchor, CheckFailure, Equipment, Person, SequenceEvent,
                      Vec3)
from . import calc, geometry as g

Vec = tuple[float, float, float]


@dataclass
class PersonSequence:
    person_id: str
    events: list[SequenceEvent] = field(default_factory=list)
    # 每站处理完事件后的挂接状态 (hookA_anchor, hookB_anchor)，失败后续为 None
    states: list[tuple[str, str] | None] = field(default_factory=list)
    failures: list[CheckFailure] = field(default_factory=list)


def _reachable_sets(stations: list[Vec], person: Person, eq: Equipment,
                    anchors: dict[str, Anchor]) -> list[set[str]]:
    return [
        {aid for aid, a in anchors.items() if calc.reachable(s, person, eq, a)}
        for s in stations
    ]


def _last_reach_table(reach: list[set[str]], anchor_ids) -> dict[str, list[int]]:
    """last[a][i] = 从站点 i 起锚点 a 连续可到达的最后一个站点索引（-1 表示 i 处不可达）。"""
    n = len(reach)
    last: dict[str, list[int]] = {}
    for a in anchor_ids:
        arr = [-1] * n
        if a in reach[n - 1]:
            arr[n - 1] = n - 1
        for i in range(n - 2, -1, -1):
            if a in reach[i]:
                arr[i] = i if a not in reach[i + 1] else arr[i + 1]
        last[a] = arr
    return last


def build_sequence(person: Person, eq: Equipment, anchors: dict[str, Anchor],
                   stations: list[Vec],
                   capacity: list[dict[str, int]]) -> PersonSequence:
    """为一名人员生成序列。capacity[i][aid] 为此前人员已在该站占用 aid 的人数，
    本函数会在成功后把本人占用写回 capacity。"""
    n = len(stations)
    res = PersonSequence(person_id=person.id)
    reach = _reachable_sets(stations, person, eq, anchors)
    last = _last_reach_table(reach, anchors.keys())

    def pos(i: int) -> Vec3:
        s = stations[i]
        return Vec3(x=s[0], y=s[1], z=s[2])

    def users_at(i: int, aid: str) -> int:
        return capacity[i].get(aid, 0)

    def candidates(i: int) -> list[str]:
        """站点 i 可挂的锚点：可到达且不超多人共用限制，按前向覆盖排序。"""
        out = []
        for a in reach[i]:
            if users_at(i, a) + 1 <= anchors[a].max_users:
                d = g.dist3(stations[i], anchors[a].position.as_tuple())
                out.append((-last[a][i], d, a))
        out.sort()
        return [a for *_x, a in out]

    def fail(i: int, action: str, msg: str, comp: dict) -> None:
        res.failures.append(CheckFailure(
            station_index=i, position=pos(i), person_id=person.id,
            action=action, check="hook_chain", message=msg, components=comp))
        res.states.extend([None] * (n - len(res.states)))

    def event(i: int, action: str, hook: str, frm, to) -> None:
        res.events.append(SequenceEvent(
            station_index=i, position=pos(i), person_id=person.id,
            action=action, hook=hook, from_anchor=frm, to_anchor=to,
            attached_after=[]))

    def refresh_attached() -> None:
        """把当前双钩状态写回最近一个事件的 attached_after。"""
        if res.events:
            res.events[-1].attached_after = sorted(
                {x for x in hooks.values() if x})

    hooks: dict[str, str | None] = {"A": None, "B": None}

    # ---- 起步挂接 ------------------------------------------------------
    c0 = candidates(0)
    if not c0:
        fail(0, "attach", "起点无可到达锚点，无法建立首个有效连接",
             {"reachable_anchors": "", "person": person.id})
        return res
    hooks["A"] = c0[0]
    event(0, "attach", "A", None, c0[0])
    refresh_attached()
    others = [a for a in c0 if a != c0[0]]
    hooks["B"] = others[0] if others else c0[0]
    event(0, "attach", "B", None, hooks["B"])
    refresh_attached()

    # ---- 沿站推进 ------------------------------------------------------
    for i in range(n):
        # 当前站不变量检查：至少一个有效连接
        live = [h for h in ("A", "B")
                if hooks[h] is not None and hooks[h] in reach[i]]
        if not live:
            fail(i, "traverse",
                 "当前站双钩均无有效连接（换挂链断开）",
                 {"hook_A": hooks["A"] or "", "hook_B": hooks["B"] or "",
                  "reachable_anchors": ",".join(sorted(reach[i]))})
            return res

        if i < n - 1:
            for h in ("A", "B"):
                cur = hooks[h]
                if cur is not None and cur not in reach[i + 1]:
                    # 该钩下一站将脱开，必须在当前站换钩（另一钩保持连接）；
                    # 换挂目标必须能覆盖到下一站，否则前行即出现双钩同时解开
                    cand = [a for a in candidates(i) if last[a][i] > i]
                    if not cand:
                        other = hooks["B" if h == "A" else "A"]
                        fail(i, "switch",
                             f"锚点 {cur} 即将超出触及范围，且当前站无锚点可换挂："
                             f"继续行进将出现双钩同时解开的瞬间",
                             {"failing_hook": h, "current_anchor": cur,
                              "other_hook_anchor": other or "",
                              "reachable_anchors": ",".join(sorted(reach[i]))})
                        return res
                    event(i, "switch", h, cur, cand[0])
                    hooks[h] = cand[0]
                    refresh_attached()

        res.states.append((hooks["A"], hooks["B"]))  # type: ignore[arg-type]
        for aid in {hooks["A"], hooks["B"]}:
            if aid:
                capacity[i][aid] = capacity[i].get(aid, 0) + 1

    # ---- 终点解钩（逐钩进行，每次事件后记录剩余连接） -------------------
    for h in ("A", "B"):
        if hooks[h] is not None:
            old = hooks[h]
            hooks[h] = None
            event(n - 1, "detach", h, old, None)
            refresh_attached()
    return res
