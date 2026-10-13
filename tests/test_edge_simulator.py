from src.edge_simulator import append_event, load_events

def test_event_csv(tmp_path):
    path=tmp_path/'events.csv'; append_event(path,'INSPECTOR-01','cable.png',.7,'REVIEW REQUIRED',12.3)
    frame=load_events(path); assert len(frame)==1 and bool(frame.iloc[0]['alert'])
