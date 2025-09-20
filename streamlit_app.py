
import streamlit as st
import numpy as np, json, hashlib, datetime, itertools, tempfile, shutil
import pandas as pd
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="PhaseU: Certificate Toolkit", layout="wide")

# ---------- GF2 helpers ----------
def gf2_rank(mat):
    M = (np.array(mat, dtype=int) % 2).astype(int)
    if M.size == 0: return 0
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i,c] == 1:
                pivot = i; break
        if pivot is None: continue
        if pivot != r: M[[r,pivot]] = M[[pivot,r]]
        for i in range(rows):
            if i!=r and M[i,c]==1: M[i] ^= M[r]
        r += 1
        if r==rows: break
    return r

def left_nullspace_basis(A):
    A = np.array(A, dtype=int) % 2
    rowsA = A.shape[0]
    M = (A.T % 2).astype(int)
    rowsM, colsM = M.shape
    R = M.copy()
    pivot_col = [-1]*rowsM
    r = 0
    for c in range(colsM):
        pivot = None
        for i in range(r, rowsM):
            if R[i,c]==1: pivot=i; break
        if pivot is None: continue
        if pivot!=r: R[[r,pivot]] = R[[pivot,r]]
        pivot_col[r] = c
        for i in range(rowsM):
            if i!=r and R[i,c]==1: R[i] ^= R[r]
        r += 1
        if r==rowsM: break
    pivot_cols = [c for c in pivot_col if c!=-1]
    free_cols = [c for c in range(colsM) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        x = np.zeros(colsM, dtype=int); x[fc]=1
        for i in range(r-1, -1, -1):
            c = pivot_col[i]; x[c] = (R[i] @ x) % 2
        basis.append(x.copy())
    # brute force fallback for small sizes
    if not basis and rowsA<=20 and colsM<=20:
        for bits in itertools.product([0,1], repeat=rowsA):
            y = np.array(bits, dtype=int)
            if np.all((A.T @ y) % 2 == 0) and np.any(y!=0):
                basis.append(y)
                break
    return basis

def choose_canonical(vecs):
    if not vecs: return None
    return min(vecs, key=lambda v:(int(np.sum(v)), tuple(v.tolist())))

# ---------- canonicalizer + verifier ----------
def canonicalize_variant(A_full, case_name=None):
    A = np.array(A_full, dtype=int) % 2
    basis = left_nullspace_basis(A)
    y = choose_canonical(basis)
    rows = A.shape[0]
    unsolvable = []
    for bits in itertools.product([0,1], repeat=rows):
        t = np.array(bits, dtype=int)
        rankA = gf2_rank(A)
        rankAug = gf2_rank(np.concatenate([A, t.reshape(-1,1)], axis=1))
        if rankAug > rankA:
            unsolvable.append(t)
    if y is None or not unsolvable:
        return None
    paired = [t for t in unsolvable if int((y@t)%2)==1]
    t = min(paired, key=lambda v:(int(np.sum(v)), tuple(v.tolist()))) if paired else min(unsolvable, key=lambda v:(int(np.sum(v)), tuple(v.tolist())))
    # compressed rep: keep columns where y·col == 1, multiplicity mod2, drop zeros
    cols = [tuple(A[:,j].tolist()) for j in range(A.shape[1])] if A.shape[1]>0 else []
    kept = {}
    for c in cols:
        dot = sum(ci*yi for ci,yi in zip(c,y)) % 2
        if dot==1: kept[c] = kept.get(c,0)+1
    kept = {c:(m%2) for c,m in kept.items() if (m%2)!=0}
    if (0,)*A.shape[0] in kept: del kept[(0,)*A.shape[0]]
    cols_list = []
    for c,m in kept.items():
        cols_list.extend([list(c)]*m)
    A_comp = np.array(cols_list).T if cols_list else np.zeros((A.shape[0],0), dtype=int)
    # rref trace
    Aug = np.concatenate([A.copy(), t.reshape(-1,1)], axis=1) if A.size>0 else np.array([[]])
    R = None
    if A.size>0:
        R = Aug.copy()%2
        rowsA, colsA = R.shape
        r=0
        for c in range(colsA):
            pivot=None
            for i in range(r, rowsA):
                if R[i,c]==1: pivot=i; break
            if pivot is None: continue
            if pivot!=r: R[[r,pivot]] = R[[pivot,r]]
            for i in range(rowsA):
                if i!=r and R[i,c]==1: R[i] ^= R[r]
            r+=1
            if r==rowsA: break
    cert = {
        "case": case_name if case_name else "variant",
        "y": y.tolist(),
        "t": t.tolist(),
        "rank_delta": int(gf2_rank(A)),
        "rank_aug": int(gf2_rank(Aug)) if A.size>0 else 0,
        "y_dot_t": int((y@t)%2),
        "A_full": A.tolist(),
        "A_comp": A_comp.tolist(),
        "compressed_cols": {str(list(k)):int(v) for k,v in kept.items()},
        "A_comp_shape": [int(A_comp.shape[0]), int(A_comp.shape[1])],
        "rref_trace_full": {"Aug_RREF": R.tolist() if R is not None else []},
        "generated_at": datetime.datetime.utcnow().isoformat()+"Z"
    }
    cert["certificate_id"] = hashlib.sha256(json.dumps(cert, sort_keys=True).encode()).hexdigest()
    return cert

def verify_against_A(cert, A_full=None):
    if A_full is None and cert.get("A_full"):
        A = np.array(cert["A_full"], dtype=int) % 2
    elif A_full is not None:
        A = np.array(A_full, dtype=int) % 2
    else:
        return {"ok": False, "error": "no A provided"}
    y = np.array(cert["y"], dtype=int); t = np.array(cert["t"], dtype=int)
    checks = {}
    checks["y_nonzero"] = bool(y.any())
    checks["yA_zero"] = bool(((y @ A) % 2 == 0).all())
    rankA = gf2_rank(A)
    Aug = np.concatenate([A, t.reshape(-1,1)], axis=1)
    rankAug = gf2_rank(Aug)
    checks["rankA"] = int(rankA); checks["rankAug"] = int(rankAug); checks["rank_increase"] = rankAug > rankA
    checks["y_dot_t"] = int((y@t)%2); checks["y_is_null"] = bool(((A.T @ y) % 2 == 0).all())
    return {"ok": all([checks["y_nonzero"], checks["yA_zero"], checks["rank_increase"], checks["y_dot_t"]==1, checks["y_is_null"]]), "checks": checks}

# ---------- variants and base matrix ----------
def add_dangling_col(A): return np.concatenate([A, np.array([0,0,0,1]).reshape(-1,1)], axis=1)
def duplicate_col0(A): return np.concatenate([A, A[:,0].reshape(-1,1)], axis=1)
def identify_col1_with_col0(A): B=A.copy(); B[:,1]=B[:,0]; return B
def add_zero_col(A): return np.concatenate([A, np.zeros((A.shape[0],1),dtype=int)], axis=1)
def remove_last_col(A): return A[:,:-1].copy() if A.shape[1]>0 else A.copy()
def flip_entries_hard_chord(A): B=A.copy(); mask=np.array([1,0,1,0]); idx=min(2,B.shape[1]-1); B[:,idx]=(B[:,idx]+mask)%2; return B
def add_sum_col2_col3(A): c2=A[:,2] if A.shape[1]>2 else np.zeros(A.shape[0],dtype=int); c3=A[:,3] if A.shape[1]>3 else np.zeros(A.shape[0],dtype=int); new=(c2+c3)%2; return np.concatenate([A,new.reshape(-1,1)],axis=1)
def append_random_col(A, seed=42): rng=np.random.RandomState(seed); col=rng.randint(0,2,size=(A.shape[0],)); return np.concatenate([A,col.reshape(-1,1)],axis=1)
def append_pair_cancel_cols(A): col=np.array([1,0,0,0]); return np.concatenate([A,col.reshape(-1,1),col.reshape(-1,1)],axis=1)

VARIANTS = {
    "T6_base": lambda A: A,
    "T6_add_dangling_col": add_dangling_col,
    "T6_duplicate_col0": duplicate_col0,
    "T6_identify_col1_with_col0": identify_col1_with_col0,
    "T6_add_zero_col": add_zero_col,
    "T6_remove_last_col": remove_last_col,
    "T6_flip_entries_hard_chord": flip_entries_hard_chord,
    "T6_add_sum_col2_col3": add_sum_col2_col3,
    "T6_append_random_col": append_random_col,
    "T6_append_pair_cancel_cols": append_pair_cancel_cols
}

T2_A = np.array([[1,1,1,0,0,0],
                 [1,0,0,0,1,1],
                 [0,0,1,1,0,1],
                 [0,1,0,1,1,0]], dtype=int)

# ---------- UI ----------
st.title("Phase U — Certificate Toolkit")
st.write("Generate canonical certificates, verify, run pilot sweeps, and export evidence bundles.")

left, right = st.columns([1,2])

with left:
    if st.button("Regenerate canonical T6 certificates (embed A_full)"):
        tmp = Path(tempfile.mkdtemp(prefix='phaseu_'))
        outdir = tmp / "certificates_v2"; outdir.mkdir()
        generated = []
        for vname, fn in VARIANTS.items():
            A_full = fn(T2_A.copy())
            cert = canonicalize_variant(A_full, case_name=vname)
            if cert is not None:
                fname = outdir / f"certificate_{vname}_{cert['certificate_id']}.json"
                json.dump(cert, open(fname, "w"), indent=2)
                generated.append(fname.name)
        st.success(f"Generated {len(generated)} certificates in {outdir}")
        st.write(generated)
        st.session_state['last_outdir'] = str(outdir)

    st.markdown('---')
    st.subheader("Upload & Verify")
    up = st.file_uploader("Upload a ZIP or JSON certificate", accept_multiple_files=False)
    if up is not None:
        upl_path = Path(tempfile.mkdtemp()) / up.name
        with open(upl_path, "wb") as f: f.write(up.getbuffer())
        if up.name.lower().endswith(".zip"):
            import zipfile
            z = zipfile.ZipFile(upl_path)
            extract_dir = upl_path.parent / "extracted"; z.extractall(extract_dir)
            cert_files = list(Path(extract_dir).rglob("*.json"))
        elif up.name.lower().endswith(".json"):
            cert_files = [upl_path]
        else:
            cert_files = []
        rows = []
        for p in cert_files:
            cert = json.load(open(p))
            res = verify_against_A(cert)
            rows.append({"file": p.name, "case": cert.get("case"), "certificate_id": cert.get("certificate_id"), "ok": res.get("ok"), **res.get("checks", {})})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
            st.download_button("Download verification CSV", df.to_csv(index=False).encode(), "verification.csv", mime="text/csv")

    st.markdown('---')
    st.subheader("Pilot sweep (T6)")
    n_trials = st.number_input("Trials per variant", min_value=1, max_value=5000, value=50, step=10)
    seed0 = st.number_input("Random seed base", value=42)
    if st.button("Run pilot sweep"):
        out_root = Path(tempfile.mkdtemp(prefix="phaseu_pilot_"))
        summary = []
        for vname, fn in VARIANTS.items():
            A0 = fn(T2_A.copy())
            for i in range(n_trials):
                seed = int(seed0) + i
                edit = np.random.RandomState(seed).choice(list(VARIANTS.keys()))
                A_pert = VARIANTS[edit](A0.copy())
                cert = canonicalize_variant(A_pert, case_name=f"{vname}_pert_{edit}")
                if cert is not None:
                    p = out_root / f"{vname}_run{i}_{cert['certificate_id']}.json"
                    json.dump(cert, open(p, "w"), indent=2)
                    res = verify_against_A(cert, A_full=np.array(cert['A_full']))
                    summary.append({"variant": vname, "trial": i, "seed": seed, "edit": edit, "cert_id": cert['certificate_id'], "ok": res.get("ok", False), **res.get("checks", {})})
                else:
                    summary.append({"variant": vname, "trial": i, "seed": seed, "edit": edit, "cert_id": None, "ok": False})
        df = pd.DataFrame(summary)
        st.success("Pilot sweep finished")
        st.dataframe(df.head(200))
        st.download_button("Download pilot CSV", df.to_csv(index=False).encode(), "pilot_summary.csv", mime="text/csv")
        st.session_state['last_pilot_df'] = df

with right:
    st.header("Workspace / Artifacts")
    if 'last_outdir' in st.session_state:
        outdir = Path(st.session_state['last_outdir'])
        st.write("Last generated certs:", outdir)
        files = [p.name for p in outdir.glob("*.json")] if outdir.exists() else []
        st.write(files)
        if st.button("Download last generated certificates (zip)") and outdir.exists():
            buf = BytesIO()
            shutil.make_archive(str(outdir), 'zip', root_dir=str(outdir))
            with open(str(outdir)+'.zip','rb') as fh:
                st.download_button("Download", fh.read(), file_name="certificates_v2.zip")

    st.markdown('---')
    if st.button("Create evidence bundle from workspace"):
        base = Path('/mnt/data')
        bundle = base / 'PhaseU_evidence_bundle_streamlit.zip'
        tmp = Path(tempfile.mkdtemp())
        items = ['consolidated_certificates_index.csv', 'verification_results_fullA_simple.json', 'certificates_v2']
        for it in items:
            p = base / it
            if p.exists():
                dst = tmp / p.name
                if p.is_dir(): shutil.copytree(p, dst)
                else: shutil.copy2(p, dst)
        shutil.make_archive(str(bundle.with_suffix('')), 'zip', root_dir=str(tmp))
        shutil.rmtree(tmp)
        if bundle.exists():
            with open(bundle,'rb') as fh:
                st.download_button("Download bundle", fh.read(), file_name=bundle.name)

st.markdown('---')
st.write("Phase U toolkit — streamlit. Ask me to add gluing/tower/plot features next.")
