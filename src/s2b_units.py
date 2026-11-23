# Stage 2 with the 256-unit head (best stage-1 width), with and without class weights.
import pickle
exec(open("s2_tune.py").read().split("res = {}")[0])
heads = pickle.load(open("outputs/s2_heads.pkl", "rb")); cw = heads["cw"]
s256 = stage1(units=256, name="S1_units256")
res = {"S1_units256": s256}
res["R9_u256"] = stage2(s256["model"], lr=1e-5, epochs=30, patience=4, units=256, name="R9_units256")
res["R10_u256_cw"] = stage2(s256["model"], lr=1e-5, epochs=30, patience=4, units=256, class_weight=cw, name="R10_units256_cw")
keep = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in res.items()}
pickle.dump(keep, open("outputs/s2b_results.pkl", "wb"))
heads["u256_head"] = s256["model"].get_weights(); pickle.dump(heads, open("outputs/s2_heads.pkl", "wb"))
print("done")
