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
                 attach_block: Callable[[int, Target, Optional[Target]],
                                        Optional[tuple[str, dict]]],
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
        # attach_block(i, target, other_hook_target) -> None 表示可挂；
        # 否则返回 (code, components)，code 为
        # duplicate_occupancy（本人另一钩已占用 / 容量被重复占用，不下结论）
        # 或 capacity_full（点锚共用上限，按旧语义判失效）。
        self.attach_block = attach_block
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

    def candidates(self, i: int, *, require_forward: bool = False,
                   other: Target | None = None) -> list[Target]:
        """站点 i 可建立连接的目标：可达、无占用阻塞、（可选）前向覆盖，
        按前向覆盖、距离、点锚优先排序。other 为本人另一钩当前目标，
        与其重复的滑梭目标不可再挂。"""
        out = []
        for t in self.reach_set(i):
            if self.attach_block(i, t, other) is not None:
                continue
            if require_forward and self.last_reach(t, i) <= i:
                continue
            kind, tid = t
            d = g.dist3(self.stations[i], self.target_pos(i, t))
            # 同前向覆盖、同距离时点锚优先（0 < 1）
            out.append((0 if kind == "anchor" else 1,
                        -self.last_reach(t, i), d, t))
        out.sort()
        return [t for *_x, t in out]

    def first_block(self, i: int, other: Target | None = None
                    ) -> tuple[Target, str, dict] | None:
        """可达目标中最早序的阻塞原因（无可挂目标时用于分类上报）。"""
        best = None
        for t in sorted(self.reach_set(i)):
            blk = self.attach_block(i, t, other)
            if blk is not None and (best is None or blk[0] < best[1]):
                best = (t, blk[0], blk[1])
        return best

    def first_forward_block(self, i: int, other: Target | None
                            ) -> tuple[Target, str, dict] | None:
        """前向（下一站仍可达）目标中的占用阻塞，用于无候选时分类上报。"""
        best = None
        for t in sorted(self.reach_set(i)):
            if self.last_reach(t, i) <= i:
                continue
            blk = self.attach_block(i, t, other)
            if blk is not None and (best is None or blk[0] < best[1]):
                best = (t, blk[0], blk[1])
        return best


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


def _jammed_open(res: PersonSequence, prov: "ReachProvider", person: Person,
                 n: int, i: int, action: str, hook: str,
                 t: Target, k: int) -> None:
    """记录 shuttle_jammed open item 并终止该人后续结论。"""
    span_id = prov.shuttle_span(t[1])
    sups = prov.span_supports(span_id)
    res.open_items.append(OpenItem(
        station_index=i, position=prov.pos_fn(i), person_id=person.id,
        action=action, code="shuttle_jammed",
        message=f"滑梭 {t[1]} 无法通过跨段 {span_id} "
                f"中间支座 {sups[k]}，滑梭将卡在支座处",
        components={"failing_hook": hook, "shuttle": t[1], "span": span_id,
                    "blocked_support_index": k,
                    "blocked_support_id": sups[k]}))
    res.states.extend([None] * (n - len(res.states)))


def _dup_occupy_open(res: PersonSequence, prov: "ReachProvider",
                     person: Person, n: int, i: int, action: str,
                     hook: str | None, comp: dict, msg: str) -> None:
    res.open_items.append(OpenItem(
        station_index=i, position=prov.pos_fn(i), person_id=person.id,
        action=action, code="duplicate_occupancy", message=msg,
        components={**comp, **({"failing_hook": hook} if hook else {})}))
    res.states.extend([None] * (n - len(res.states)))


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
        blk = prov.first_block(0)
        if blk is not None and blk[1] == "duplicate_occupancy":
            t, _code, comp = blk
            res.open_items.append(OpenItem(
                station_index=0, position=pos(0), person_id=person.id,
                action="attach", code="duplicate_occupancy",
                message=f"滑梭 {t[1]} 已被占用，双钩重复挂接同一滑梭"
                        f"（重复占用），不下结论",
                components=comp))
            res.states.extend([None] * n)
            return res
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
    # B 钩不得与 A 钩重复挂同一滑梭；重复占用即不下结论
    cB = prov.candidates(0, other=hooks["A"])
    if not cB:
        blk = prov.first_block(0, other=hooks["A"])
        if blk is not None and blk[1] == "duplicate_occupancy":
            t, _code, comp = blk
            res.open_items.append(OpenItem(
                station_index=0, position=pos(0), person_id=person.id,
                action="attach", code="duplicate_occupancy",
                message=f"B 钩只能与 A 钩重复挂到滑梭 {t[1]}"
                        f"（重复占用），双钩不独立，不下结论",
                components={**comp, "failing_hook": "B"}))
            res.states.extend([None] * n)
            return res
        # 无其他可达目标：单钩起步继续（自动算法按单连接推进）
        hooks["B"] = None
    else:
        hooks["B"] = cB[0]
        res.events.append(_ev(0, pos(0), person.id, "attach", "B", None,
                              hooks["B"], hooks))
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
                        _jammed_open(res, prov, person, n, i, "traverse",
                                     h, cur, k)
                        return res
                if prov.reachable(i + 1, cur):
                    continue
                # 该钩下一站将脱开，必须当前站换钩（另一钩保持连接）；
                # 目标不得与另一钩重复占用同一滑梭
                other = hooks["B" if h == "A" else "A"]
                cand = prov.candidates(i, require_forward=True, other=other)
                if not cand:
                    blk = prov.first_forward_block(i, other)
                    if blk is not None and blk[1] == "duplicate_occupancy":
                        bt, _code, bcomp = blk
                        _dup_occupy_open(
                            res, prov, person, n, i, "switch", h, bcomp,
                            f"该钩只能重复挂到另一钩已占用的滑梭 {bt[1]}"
                            f"（重复占用），双钩不独立，不下结论")
                        return res
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
        for h in ("A", "B"):
            if hooks[h]:
                prov.occupy(i, hooks[h], h)

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

    # 每站处理完动作后实际生效的挂接（用于过支座检查；不依赖中间态）
    attached_at: list[dict[str, Target | None]] = []

    for i in range(n):
        # 进站：上一站已挂的滑梭若随人员越过不可通过的中间支座 → 卡支座
        if attached_at:
            prev = attached_at[-1]
            for h in ("A", "B"):
                cur = prev.get(h)
                if cur is not None and cur[0] == "shuttle":
                    k = prov.crosses_blocked(cur[1], i - 1, i)
                    if k is not None:
                        _jammed_open(res, prov, person, n, i, "traverse",
                                     h, cur, k)
                        return res

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
                other_t = hooks["B" if h == "A" else "A"]
                blk = prov.attach_block(i, tgt, other_t)
                if blk is not None:
                    code, bcomp = blk
                    if code == "duplicate_occupancy":
                        if tgt[0] == "shuttle" and other_t == tgt:
                            msg = (f"A/B 钩在同一站重复挂到同一滑梭 {tgt[1]}"
                                   f"（重复占用），双钩不独立，不下结论")
                        else:
                            msg = (f"滑梭 {tgt[1]} 已被占满，重复占用，"
                                   f"无法由本核算判定")
                        _dup_occupy_open(res, prov, person, n, i,
                                         act.action, h, bcomp, msg)
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
        attached_at.append(dict(hooks))
        for h in ("A", "B"):
            if hooks[h]:
                prov.occupy(i, hooks[h], h)

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
