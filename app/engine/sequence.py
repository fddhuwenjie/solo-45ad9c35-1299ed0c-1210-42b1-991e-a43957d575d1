"""挂接、换钩与解钩序列生成：双钩交替前进，任何状态至少保留一个有效连接。

挂接目标统一为（kind, id）：kind='anchor' 为固定点锚，kind='shuttle' 为
柔性跨段滑梭。每站每个目标的可达性、共用容量与滑梭跨支座信息由
ReachProvider 提供；滑梭跨越不允许通过的中间支座记 shuttle_jammed
open item（不下结论）。柔性体系出现无目标可续的瞬间记 continuity_break
（不下结论）；纯点锚问题沿用 hook_chain 失效判定，保持旧请求兼容。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..models import (CheckFailure, Equipment, ManualHookAction, OpenItem,
                      Person, SequenceEvent, Vec3)
from . import geometry as g

Vec = tuple[float, float, float]
Target = tuple[str, str]          # ('anchor'|'shuttle', id)


@dataclass
class PersonSequence:
    person_id: str
    events: list[SequenceEvent] = field(default_factory=list)
    # 每站处理完事件后的挂接状态 ((kind,id)|None, (kind,id)|None)；失败后为 None
    states: list[tuple[Optional[Target], Optional[Target]] | None] = field(default_factory=list)
    failures: list[CheckFailure] = field(default_factory=list)
    open_items: list[OpenItem] = field(default_factory=list)


class ReachProvider:
    """提供每站目标可达性、前向覆盖、共用容量、排序位置与滑梭跨支座信息。"""

    def __init__(self, n: int, stations: list[Vec],
                 reachable: Callable[[int, Target], bool],
                 last_reach: Callable[[Target, int], int],
                 capacity_ok: Callable[[int, Target], bool],
                 occupy: Callable[[int, Target], None],
                 targets: list[Target],
                 pos_fn: Callable[[int], Vec3],
                 target_pos: Callable[[int, Target], Vec],
                 anchor_max_users: Callable[[str], int],
                 shuttle_span: Callable[[str], str] | None = None,
                 span_supports: Callable[[str], list[str]] | None = None,
                 crosses_blocked: Callable[[str, int, int], int | None] | None = None):
        self.n = n
        self.stations = stations
        self.reachable = reachable
        self.last_reach = last_reach
        self.capacity_ok = capacity_ok
        self.occupy = occupy
        self.targets = targets
        self.pos_fn = pos_fn
        self.target_pos = target_pos
        self.anchor_max_users = anchor_max_users
        self.shuttle_span = shuttle_span or (lambda sid: "")
        self.span_supports = span_supports or (lambda sid: [])
        self.crosses_blocked = crosses_blocked or (lambda sid, i0, i1: None)

    def reach_set(self, i: int) -> set[Target]:
        return {t for t in self.targets if self.reachable(i, t)}

    def candidates(self, i: int, *, require_forward: bool = False) -> list[Target]:
        """站点 i 可建立连接的目标：可达且不超共用限制，按前向覆盖、距离排序。"""
        out = []
        for t in self.reach_set(i):
            if not self.capacity_ok(i, t):
                continue
            if require_forward and self.last_reach(t, i) <= i:
                continue
            kind, tid = t
            d = g.dist3(self.stations[i], self.target_pos(i, t))
            out.append((-self.last_reach(t, i), d, 0 if kind == "anchor" else 1, t))
        out.sort()
        return [t for *_x, t in out]


def _ev(station_index: int, pos: Vec3, person_id: str, action: str, hook: str,
        frm: Target | None, to: Target | None, hooks: dict) -> SequenceEvent:
    def part(t: Target | None, kind: str):
        return t[1] if t is not None and t[0] == kind else None
    return SequenceEvent(
        station_index=station_index, position=pos, person_id=person_id,
        action=action, hook=hook,
        from_anchor=part(frm, "anchor"), to_anchor=part(to, "anchor"),
        from_shuttle=part(frm, "shuttle"), to_shuttle=part(to, "shuttle"),
        attached_after=sorted({t[1] for t in hooks.values()
                               if t is not None and t[0] == "anchor"}),
        attached_shuttles_after=sorted(
            {t[1] for t in hooks.values() if t is not None and t[0] == "shuttle"}))


def _tid(t: Target | None) -> str:
    if t is None:
        return ""
    return ("anchor:" if t[0] == "anchor" else "shuttle:") + t[1]


def _id_kind(t: Target | None, kind: str) -> str:
    return t[1] if t is not None and t[0] == kind else ""


def _fmt(ts) -> str:
    return ",".join(f"{k}:{x}" for k, x in sorted(ts))


def _tgt_name(t: Target | None) -> str:
    if t is None:
        return ""
    return f"{'滑梭' if t[0] == 'shuttle' else '锚点'} {t[1]}"


def _has_flex(hooks: dict) -> bool:
    return any(t is not None and t[0] == "shuttle" for t in hooks.values())


def build_sequence(person: Person, eq: Equipment, prov: ReachProvider
                   ) -> PersonSequence:
    """为一名人员自动生成双钩序列（点锚 + 滑梭混合）。"""
    n = prov.n
    res = PersonSequence(person_id=person.id)

    def pos(i: int) -> Vec3:
        return prov.pos_fn(i)

    def refresh():
        if res.events:
            ev = res.events[-1]
            ev.attached_after = sorted(
                {t[1] for t in hooks.values()
                 if t is not None and t[0] == "anchor"})
            ev.attached_shuttles_after = sorted(
                {t[1] for t in hooks.values()
                 if t is not None and t[0] == "shuttle"})

    hooks: dict[str, Target | None] = {"A": None, "B": None}

    def live(i: int) -> list[str]:
        return [h for h in ("A", "B")
                if hooks[h] is not None and prov.reachable(i, hooks[h])]

    def break_or_fail(i: int, action: str, comp: dict, msg_anchor: str,
                      msg_flex: str, hook=None):
        """无目标可续：体系含滑梭 → continuity_break（不下结论）；纯点锚 → 失效。"""
        flex = _has_flex(hooks) or any(
            t[0] == "shuttle" for t in prov.reach_set(i))
        if flex:
            res.open_items.append(OpenItem(
                station_index=i, position=pos(i), person_id=person.id,
                action=action, code="continuity_break", message=msg_flex,
                components={**comp, **({"failing_hook": hook} if hook else {})}))
        else:
            res.failures.append(CheckFailure(
                station_index=i, position=pos(i), person_id=person.id,
                action=action, check="hook_chain", message=msg_anchor,
                components={**comp, **({"failing_hook": hook} if hook else {})}))
        res.states.extend([None] * (n - len(res.states)))

    # ---- 起步挂接 ------------------------------------------------------
    c0 = prov.candidates(0)
    if not c0:
        reach = prov.reach_set(0)
        if reach or any(t[0] == "shuttle" for t in prov.targets):
            break_or_fail(
                0, "attach",
                {"reachable_targets": _fmt(reach)},
                "起点无可到达锚点，无法建立首个有效连接",
                "起点无法在柔性跨段上建立有效连接（连续性断开），"
                "是否可通行无法由本核算判定")
        else:
            res.failures.append(CheckFailure(
                station_index=0, position=pos(0), person_id=person.id,
                action="attach", check="hook_chain",
                message="起点无可到达锚点，无法建立首个有效连接",
                components={"reachable_anchors": "", "person": person.id}))
            res.states.extend([None] * n)
        return res
    hooks["A"] = c0[0]
    res.events.append(_ev(0, pos(0), person.id, "attach", "A", None, c0[0], hooks))
    refresh()
    others = [t for t in c0 if t != c0[0]]
    hooks["B"] = others[0] if others else c0[0]
    res.events.append(_ev(0, pos(0), person.id, "attach", "B", None, hooks["B"], hooks))
    refresh()

    # ---- 沿站推进 ------------------------------------------------------
    for i in range(n):
        if not live(i):
            comp = {"hook_A": _tid(hooks["A"]), "hook_B": _tid(hooks["B"]),
                    "reachable_targets": _fmt(prov.reach_set(i))}
            if _has_flex(hooks):
                res.open_items.append(OpenItem(
                    station_index=i, position=pos(i), person_id=person.id,
                    action="traverse", code="continuity_break",
                    message="行进至当前站时已无有效连接，柔性体系连续性断开，"
                            "无法判定是否可通行", components=comp))
            else:
                res.failures.append(CheckFailure(
                    station_index=i, position=pos(i), person_id=person.id,
                    action="traverse", check="hook_chain",
                    message="当前站双钩均无有效连接（换挂链断开）",
                    components=comp))
            res.states.extend([None] * (n - len(res.states)))
            return res

        if i < n - 1:
            for h in ("A", "B"):
                cur = hooks[h]
                if cur is None:
                    continue
                # 滑梭随人员移动：越过不允许通过的中间支座 → 卡支座
                if cur[0] == "shuttle":
                    k = prov.crosses_blocked(cur[1], i, i + 1)
                    if k is not None:
                        span_id = prov.shuttle_span(cur[1])
                        sups = prov.span_supports(span_id)
                        res.open_items.append(OpenItem(
                            station_index=i, position=pos(i),
                            person_id=person.id, action="traverse",
                            code="shuttle_jammed",
                            message=f"滑梭 {cur[1]} 无法通过跨段 {span_id} "
                                    f"中间支座 {sups[k]}，滑梭将卡在支座处",
                            components={"failing_hook": h, "shuttle": cur[1],
                                        "span": span_id,
                                        "blocked_support_index": k,
                                        "blocked_support_id": sups[k]}))
                        res.states.extend([None] * (n - len(res.states)))
                        return res
                if prov.reachable(i + 1, cur):
                    continue
                # 该钩下一站将脱开，必须当前站换钩（另一钩保持连接）
                cand = prov.candidates(i, require_forward=True)
                if not cand:
                    other = hooks["B" if h == "A" else "A"]
                    comp = {"current_anchor": _id_kind(cur, "anchor"),
                            "current_target": _tid(cur),
                            "other_hook_target": _tid(other),
                            "reachable_targets": _fmt(prov.reach_set(i))}
                    break_or_fail(
                        i, "switch", comp,
                        f"锚点 {cur[1]} 即将超出触及范围，且当前站无锚点可换挂："
                        f"继续行进将出现双钩同时解开的瞬间",
                        f"挂接目标 {_tgt_name(cur)} 即将超出触及范围，当前站"
                        f"无可续目标，柔性体系连续性断开，无法判定", h)
                    return res
                tgt = cand[0]
                res.events.append(_ev(i, pos(i), person.id, "switch", h, cur, tgt, hooks))
                hooks[h] = tgt
                refresh()

        res.states.append((hooks["A"], hooks["B"]))
        for t in {hooks["A"], hooks["B"]}:
            if t:
                prov.occupy(i, t)

    # ---- 终点解钩 ------------------------------------------------------
    for h in ("A", "B"):
        if hooks[h] is not None:
            old = hooks[h]
            hooks[h] = None
            res.events.append(_ev(n - 1, pos(n - 1), person.id, "detach", h, old, None, hooks))
            refresh()
    return res


def replay_sequence(person: Person, eq: Equipment, prov: ReachProvider,
                    actions: list[ManualHookAction]) -> PersonSequence:
    """按人工给定次序回放挂接动作并逐站校验（目标可为点锚或滑梭）。"""
    n = prov.n
    res = PersonSequence(person_id=person.id)

    def pos(i: int) -> Vec3:
        return prov.pos_fn(i)

    def refresh():
        if res.events:
            ev = res.events[-1]
            ev.attached_after = sorted(
                {t[1] for t in hooks.values()
                 if t is not None and t[0] == "anchor"})
            ev.attached_shuttles_after = sorted(
                {t[1] for t in hooks.values()
                 if t is not None and t[0] == "shuttle"})

    def hook_fail(i, action, msg, comp, hook=None):
        res.failures.append(CheckFailure(
            station_index=i, position=pos(i), person_id=person.id,
            action=action, check="hook_chain", message=msg,
            components={**comp, **({"failing_hook": hook} if hook else {})}))
        res.states.extend([None] * (n - len(res.states)))

    by_station: dict[int, list[ManualHookAction]] = {}
    for act in actions:
        by_station.setdefault(act.station_index, []).append(act)

    hooks: dict[str, Target | None] = {"A": None, "B": None}

    def live(i: int) -> list[str]:
        return [h for h in ("A", "B")
                if hooks[h] is not None and prov.reachable(i, hooks[h])]

    for i in range(n):
        if any(hooks.values()) and not live(i):
            comp = {"hook_A": _tid(hooks["A"]), "hook_B": _tid(hooks["B"]),
                    "reachable_targets": _fmt(prov.reach_set(i))}
            if _has_flex(hooks):
                res.open_items.append(OpenItem(
                    station_index=i, position=pos(i), person_id=person.id,
                    action="traverse", code="continuity_break",
                    message="行进至当前站时已无有效连接，柔性体系连续性断开",
                    components=comp))
            else:
                hook_fail(i, "traverse",
                          "行进至当前站时已无有效连接（换挂链断开）", comp)
            return res

        for act in by_station.get(i, []):
            h = act.hook
            if act.action == "detach":
                other = "B" if h == "A" else "A"
                if hooks[other] is None or not prov.reachable(i, hooks[other]):
                    hook_fail(i, "detach",
                             f"解钩 {h} 后另一钩无有效连接，不允许松开",
                             {"other_hook_anchor":
                                  _id_kind(hooks[other], "anchor"),
                              "other_hook_shuttle":
                                  _id_kind(hooks[other], "shuttle")}, h)
                    return res
                old = hooks[h]
                hooks[h] = None
                res.events.append(_ev(i, pos(i), person.id, "detach", h, old, None, hooks))
            else:
                tgt: Target = ("shuttle", act.shuttle) if act.shuttle \
                    else ("anchor", act.anchor)
                if not prov.reachable(i, tgt):
                    hook_fail(i, act.action,
                              f"目标 {_tgt_name(tgt)} 在当前站超出连接器触及范围",
                              {"target_anchor": act.anchor or "",
                               "target_shuttle": act.shuttle or "",
                               "reachable_targets": _fmt(prov.reach_set(i))}, h)
                    return res
                if not prov.capacity_ok(i, tgt):
                    if tgt[0] == "shuttle":
                        res.open_items.append(OpenItem(
                            station_index=i, position=pos(i),
                            person_id=person.id, action=act.action,
                            code="duplicate_occupancy",
                            message=f"滑梭 {tgt[1]} 容量已满，重复占用"
                                    f"（同梭重复挂接），无法由本核算判定",
                            components={"failing_hook": h,
                                        "target_shuttle": tgt[1]}))
                        res.states.extend([None] * (n - len(res.states)))
                    else:
                        cap = prov.anchor_max_users(tgt[1])
                        hook_fail(i, act.action,
                                 f"锚点 {tgt[1]} 共用人数已达上限 {cap}",
                                 {"target_anchor": tgt[1], "max_users": cap}, h)
                    return res
                old = hooks[h]
                if old is not None:
                    other = "B" if h == "A" else "A"
                    if hooks[other] is None or not prov.reachable(i, hooks[other]):
                        comp = {"from_anchor": _id_kind(old, "anchor"),
                                "from_shuttle": _id_kind(old, "shuttle"),
                                "target_anchor": act.anchor or "",
                                "target_shuttle": act.shuttle or "",
                                "other_hook_anchor":
                                    _id_kind(hooks[other], "anchor"),
                                "other_hook_shuttle":
                                    _id_kind(hooks[other], "shuttle"),
                                "reachable_targets": _fmt(prov.reach_set(i))}
                        if _has_flex(hooks) or old[0] == "shuttle" or tgt[0] == "shuttle":
                            res.open_items.append(OpenItem(
                                station_index=i, position=pos(i),
                                person_id=person.id, action=act.action,
                                code="continuity_break",
                                message=f"换挂 {h} 钩期间另一钩无有效连接：旧连接 "
                                        f"{old[1]} 断开至新连接 {tgt[1]} 建立前"
                                        f"柔性体系连续性断开，无法判定",
                                components={**comp, "failing_hook": h}))
                            res.states.extend([None] * (n - len(res.states)))
                        else:
                            hook_fail(i, act.action,
                                     f"换挂 {h} 钩期间另一钩无有效连接：旧连接 "
                                     f"{old[1]} 断开至新连接 {tgt[1]} 建立前"
                                     f"将失去全部有效连接", comp, h)
                        return res
                hooks[h] = tgt
                res.events.append(_ev(i, pos(i), person.id, act.action, h, old, tgt, hooks))
            refresh()

        if i < n - 1 and not live(i):
            comp = {"hook_A": _tid(hooks["A"]), "hook_B": _tid(hooks["B"])}
            if _has_flex(hooks):
                res.open_items.append(OpenItem(
                    station_index=i, position=pos(i), person_id=person.id,
                    action="traverse", code="continuity_break",
                    message="当前站结束后无任何有效连接，柔性体系连续性断开",
                    components=comp))
            else:
                hook_fail(i, "traverse",
                          "当前站结束后无任何有效连接，"
                          "继续行进将出现双钩同时解开的瞬间", comp)
            return res

        res.states.append((hooks["A"], hooks["B"]))
        for t in {hooks["A"], hooks["B"]}:
            if t:
                prov.occupy(i, t)

    leftover = [a for kk, acts in by_station.items() if kk >= n for a in acts]
    if leftover:
        a0 = leftover[0]
        hook_fail(n - 1, a0.action,
                  f"人工动作站点 {a0.station_index} 超出路线站点范围（共 {n} 站）",
                  {"station_index": a0.station_index, "station_count": n})
        return res

    for h in ("A", "B"):
        if hooks[h] is not None:
            old = hooks[h]
            hooks[h] = None
            res.events.append(_ev(n - 1, pos(n - 1), person.id, "detach", h, old, None, hooks))
            refresh()
    return res
