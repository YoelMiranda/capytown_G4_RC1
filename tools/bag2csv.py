import sys, math, csv
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
ts = get_typestore(Stores.ROS2_HUMBLE)
def yaw(q): return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
for bag in sys.argv[2:]:
    out = Path(sys.argv[1]) / (Path(bag).name + "_desde_bag.csv")
    with AnyReader([Path(bag)], default_typestore=ts) as r, open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t","topic","x","y","yaw","v","w"])
        for c, t, raw in r.messages(connections=[c for c in r.connections if c.topic in ("/odom","/cmd_vel")]):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == "/odom":
                p = m.pose.pose; tw = m.twist.twist
                w.writerow([t/1e9,"odom",p.position.x,p.position.y,yaw(p.orientation),tw.linear.x,tw.angular.z])
            else:
                w.writerow([t/1e9,"cmd","","","",m.linear.x,m.angular.z])
    print("ok", out)
