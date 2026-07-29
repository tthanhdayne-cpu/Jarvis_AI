"""Mock-first tests and opt-in live diagnostics for Chrome background tabs."""
from __future__ import annotations
import argparse, json, threading, unittest
from actions.browser_tab_actions import BrowserTabService
from actions.browser_tab_bridge import NamedPipeBridge

class Runtime:
    def __init__(self):
        self.state="ACTIVE"; self.sleep=False; self.generation=4; self.expected_generation=4
        self.cancellation_event=threading.Event(); self.source_turn=10
        self.state_getter=lambda:self.state; self.sleep_intent_getter=lambda:self.sleep
        self.generation_getter=lambda:self.generation

class FakeTransport:
    def __init__(self):
        self.connection_id="conn-1"; self.profile="profile-1"; self.extension="ext-1"
        self.tabs=[{"id":101,"window_id":1,"index":0,"title":"YouTube — Chicago","hostname":"www.youtube.com","active":True,"pinned":False},
                   {"id":102,"window_id":1,"index":1,"title":"Gmail Inbox","hostname":"mail.google.com","active":False,"pinned":False},
                   {"id":103,"window_id":1,"index":2,"title":"YouTube Studio","hostname":"studio.youtube.com","active":False,"pinned":False}]
        for tab in self.tabs:
            tab.setdefault("audible",False); tab.setdefault("muted",False)
        self.calls=[]; self.tokens={}; self.late=False
    def request(self, action, arguments, runtime):
        self.calls.append((action,dict(arguments)))
        base={"request_id":"r","connection_id":self.connection_id,"session_generation":runtime.expected_generation,
              "success":True,"status":"completed","error":None}
        if action=="list_tabs": self.tokens[arguments["snapshot_token"]]=set(t["id"] for t in self.tabs)
        if action in {"list_tabs","validate_tabs"}:
            ids=arguments.get("tab_ids")
            tabs=[dict(t) for t in self.tabs if ids is None or t["id"] in ids]
            base["data"]={"tabs":tabs,"extension_id":self.extension,"profile_instance_id":self.profile}
            return base
        if action in {"close_tab","close_tabs"}:
            ids=arguments["tab_ids"]; self.tabs=[t for t in self.tabs if t["id"] not in ids]
            base["data"]={"closed_count":len(ids),"extension_id":self.extension,"profile_instance_id":self.profile}; return base
        if action=="close_duplicate_tabs":
            ids=arguments["tab_ids"]; self.tabs=[t for t in self.tabs if t["id"] not in ids]
            base["data"]={"closed_count":len(ids),"extension_id":self.extension,"profile_instance_id":self.profile}; return base
        if action=="create_tab": base["data"]={"created":True,"hostname":"example.com"}; return base
        if action in {"focus_tab","reload_tab","mute_tab","unmute_tab","pin_tab","unpin_tab"}:
            ids=arguments["tab_ids"]
            for tab in self.tabs:
                if tab["id"] in ids:
                    if action=="focus_tab": tab["active"]=True
                    if action=="mute_tab": tab["muted"]=True
                    if action=="unmute_tab": tab["muted"]=False
                    if action=="pin_tab": tab["pinned"]=True
                    if action=="unpin_tab": tab["pinned"]=False
            base["data"]={"affected_count":len(ids),"extension_id":self.extension,"profile_instance_id":self.profile}; return base
        if action=="ping": base["data"]={"pong":True}; return base
        return {**base,"success":False,"status":"unknown_command","data":{}}
    def invalidate(self): pass
    def close(self): pass

class BrowserTabTests(unittest.TestCase):
    def setUp(self):
        self.transport=FakeTransport(); self.audit=[]; self.now=[100.0]; self.runtime=Runtime()
        self.service=BrowserTabService(self.transport,audit=lambda **r:self.audit.append(r),clock=lambda:self.now[0])
    def approve(self): return self.service.resolve_voice("Xác nhận",turn_id=11,runtime=self.runtime)
    def test_ping(self):
        self.assertTrue(self.transport.request("ping",{},self.runtime)["data"]["pong"])
    def test_one_match_pending_no_close(self):
        r=self.service.close_by_title(title_query="Gmail",runtime=self.runtime)
        self.assertEqual(r["status"],"confirmation_required"); self.assertFalse(any(c[0].startswith("close_") for c in self.transport.calls))
    def test_many_matches_clarification(self):
        r=self.service.close_by_title(title_query="YouTube",runtime=self.runtime)
        self.assertEqual(r["status"],"clarification_required"); self.assertEqual(len(r["data"]["options"]),2)
    def test_zero_match(self): self.assertEqual(self.service.close_by_title(title_query="none",runtime=self.runtime)["status"],"tab_not_found")
    def test_pinned_blocked(self):
        self.transport.tabs[1]["pinned"]=True
        self.assertEqual(self.service.close_by_title(title_query="Gmail",runtime=self.runtime)["status"],"pinned_tab_blocked")
    def test_choice_then_confirmation(self):
        r=self.service.close_by_title(title_query="YouTube",runtime=self.runtime); idx=r["data"]["options"][1]["index"]
        self.assertEqual(self.service.close_by_index(index=idx,runtime=self.runtime)["status"],"confirmation_required")
    def test_title_changed(self):
        self.service.close_by_title(title_query="Gmail",runtime=self.runtime); self.transport.tabs[1]["title"]="Changed"
        self.assertEqual(self.approve()["status"],"snapshot_changed")
    def test_hostname_changed(self):
        self.service.close_by_title(title_query="Gmail",runtime=self.runtime); self.transport.tabs[1]["hostname"]="example.com"
        self.assertEqual(self.approve()["status"],"snapshot_changed")
    def test_tab_disappeared(self):
        self.service.close_by_title(title_query="Gmail",runtime=self.runtime); self.transport.tabs.pop(1)
        self.assertEqual(self.approve()["status"],"tab_not_found")
    def test_tab_set_changed(self):
        self.service.close_all(runtime=self.runtime,title_query="YouTube"); self.transport.tabs.pop()
        self.assertEqual(self.approve()["status"],"tab_set_changed")
    def test_sleep_cancel(self):
        self.service.close_by_title(title_query="Gmail",runtime=self.runtime); self.runtime.sleep=True; self.runtime.cancellation_event.set()
        self.assertEqual(self.approve()["status"],"cancelled")
    def test_generation_mismatch(self):
        self.service.close_by_title(title_query="Gmail",runtime=self.runtime); self.runtime.generation=5
        self.assertEqual(self.approve()["status"],"stale_session")
    def test_connection_mismatch(self):
        self.service.close_by_title(title_query="Gmail",runtime=self.runtime); self.transport.connection_id="conn-2"
        self.assertEqual(self.approve()["status"],"connection_mismatch")
    def test_profile_mismatch(self):
        self.service.close_by_title(title_query="Gmail",runtime=self.runtime); self.transport.profile="profile-2"
        self.assertEqual(self.approve()["status"],"profile_mismatch")
    def test_index_latest_and_expiry(self):
        listed=self.service.list_tabs(runtime=self.runtime); idx=listed["data"]["tabs"][0]["index"]
        self.assertEqual(self.service.close_by_index(index=idx,runtime=self.runtime)["status"],"confirmation_required")
        self.service.cancel(); self.now[0]=121
        self.assertEqual(self.service.close_by_index(index=idx,runtime=self.runtime)["status"],"stale_tab_list")
    def test_close_many_once(self):
        self.service.close_all(runtime=self.runtime,title_query="YouTube")
        self.assertEqual(self.approve()["data"]["closed_count"],2)
        self.assertEqual(sum(1 for c in self.transport.calls if c[0]=="close_tabs"),1)
    def test_raw_id_not_public_or_audit(self):
        result=self.service.list_tabs(runtime=self.runtime); text=json.dumps(result)+json.dumps(self.audit)
        self.assertNotIn('101',text); self.assertNotIn('"id"',text)
    def test_unknown_command_blocked(self):
        bridge=NamedPipeBridge(); runtime=self.runtime
        self.assertEqual(bridge.request("arbitrary_js",{},runtime)["status"],"unknown_command")
    def test_focus_reload_mute_pin(self):
        self.assertEqual(self.service.focus_tab(title_query="Gmail",runtime=self.runtime)["status"],"completed")
        self.assertEqual(self.service.reload_tab(title_query="Gmail",runtime=self.runtime)["status"],"completed")
        self.assertEqual(self.service.mute_tab(title_query="Gmail",runtime=self.runtime)["status"],"completed")
        self.assertEqual(self.service.unmute_tab(title_query="Gmail",runtime=self.runtime)["status"],"completed")
        self.assertEqual(self.service.pin_tab(title_query="Gmail",runtime=self.runtime)["status"],"completed")
        self.assertEqual(self.service.unpin_tab(title_query="Gmail",runtime=self.runtime)["status"],"completed")
    def test_audible_multiple_requires_clarification(self):
        self.transport.tabs[0]["audible"]=True; self.transport.tabs[2]["audible"]=True
        self.assertEqual(self.service.mute_tab(target="audible",runtime=self.runtime)["status"],"clarification_required")
    def test_open_tab(self):
        self.assertEqual(self.service.open_tab(url="https://example.com",runtime=self.runtime)["status"],"completed")
    def test_safe_action_multiple_match_clarifies(self):
        self.assertEqual(self.service.focus_tab(title_query="YouTube",runtime=self.runtime)["status"],"clarification_required")
    def test_duplicate_close_requires_confirmation(self):
        self.transport.tabs.append(dict(self.transport.tabs[1],id=104,index=3))
        result=self.service.close_duplicates(runtime=self.runtime)
        self.assertEqual(result["status"],"confirmation_required")
        self.assertFalse(any(call[0]=="close_duplicate_tabs" for call in self.transport.calls))
        self.assertEqual(self.approve()["status"],"completed")
    def test_late_response_discarded(self):
        original=self.transport.request
        def late(action,arguments,runtime):
            result=original(action,arguments,runtime); runtime.generation+=1; return result
        self.transport.request=late
        self.assertEqual(self.service.focus_tab(title_query="Gmail",runtime=self.runtime)["status"],"stale_session")
    def test_public_shape_has_no_raw_chrome_index(self):
        result=self.service.list_tabs(runtime=self.runtime); tab=result["data"]["tabs"][0]
        self.assertEqual(set(tab),{"index","tab_ref","title","hostname","active","pinned","audible","muted"})
        self.assertNotIn("chrome_index",json.dumps(result))
    def test_registry_schema_has_no_raw_tab_id(self):
        from actions.windows_action_registry import ACTION_DEFINITIONS
        schemas=json.dumps([d.parameter_schema for d in ACTION_DEFINITIONS])
        self.assertNotIn("tab_id",schemas); self.assertNotIn("tab_ids",schemas)

def live_main(argv):
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["ping","list","close-title"])
    parser.add_argument("--title"); parser.add_argument("--live",action="store_true"); args=parser.parse_args(argv)
    runtime=Runtime(); service=BrowserTabService()
    if args.command=="ping": print(service.transport.request("ping",{},runtime)); return
    if args.command=="list": print(json.dumps(service.list_tabs(runtime=runtime),ensure_ascii=False,indent=2)); return
    if not args.live: print("Refusing to close without --live"); return
    pending=service.close_by_title(title_query=args.title or "",runtime=runtime); print(json.dumps(pending,ensure_ascii=False,indent=2))
    if pending.get("status")!="confirmation_required": return
    if input("Type Xác nhận to continue: ").strip()!="Xác nhận": print("Cancelled"); return
    print(service.resolve_voice("Xác nhận",turn_id=11,runtime=runtime))

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1] in {"ping","list","close-title"}: live_main(sys.argv[1:])
    else: unittest.main()
