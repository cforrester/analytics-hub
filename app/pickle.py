# app/pickle.py

import streamlit as st
import io
import pickle
import pickletools
import hashlib
import pandas as pd
import importlib
import sys

# --- Security: Restricted Unpickler with whitelist and module aliasing ---
class RestrictedUnpickler(pickle.Unpickler):
    SAFE_BUILTINS = {
        'builtins': {
            'dict', 'list', 'set', 'tuple',
            'str', 'bytes', 'int', 'float',
            'bool', 'NoneType'
        }
    }
    SAFE_GLOBALS = {
        'sklearn.linear_model.logistic':    {'LogisticRegression', 'LogisticRegressionCV'},
        'sklearn.linear_model._logistic':   {'LogisticRegression', 'LogisticRegressionCV'},
        'sklearn.ensemble._forest':         {'RandomForestClassifier', 'RandomForestRegressor'},
        'sklearn.tree._classes':            {'DecisionTreeClassifier', 'DecisionTreeRegressor'},
        'numpy.core.multiarray':            {'_reconstruct'},
        'numpy':                            {'ndarray', 'dtype'},
    }
    MODULE_ALIASES = {
        'sklearn.linear_model.logistic': 'sklearn.linear_model._logistic'
    }

    def find_class(self, module, name):
        if module in self.SAFE_BUILTINS and name in self.SAFE_BUILTINS[module]:
            return getattr(__builtins__, name)
        allowed = self.SAFE_GLOBALS.get(module)
        if allowed and name in allowed:
            real_mod = self.MODULE_ALIASES.get(module, module)
            try:
                m = importlib.import_module(real_mod)
            except ModuleNotFoundError:
                raise pickle.UnpicklingError(f"Cannot import whitelisted module '{real_mod}'")
            return getattr(m, name)
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

def safe_loads(data: bytes):
    for old_mod, real_mod in RestrictedUnpickler.MODULE_ALIASES.items():
        if old_mod not in sys.modules:
            try:
                sys.modules[old_mod] = importlib.import_module(real_mod)
            except ImportError:
                pass
    return RestrictedUnpickler(io.BytesIO(data)).load()

# --- Feature handlers ---

def _metadata(pickle_bytes):
    md5    = hashlib.md5(pickle_bytes).hexdigest()
    sha256 = hashlib.sha256(pickle_bytes).hexdigest()
    st.write(f"**MD5:** `{md5}`")
    st.write(f"**SHA-256:** `{sha256}`")
    protos = [arg for op, arg, _ in pickletools.genops(pickle_bytes) if op.name=='PROTO']
    st.write(f"**Protocol version:** {protos[0] if protos else 'unknown'}")

def _disassemble(pickle_bytes):
    st.code(pickletools.dis(pickle_bytes, annotate=True), language='text')

def _list_globals(pickle_bytes):
    parsed = []
    for op, arg, _ in pickletools.genops(pickle_bytes):
        if op.name=='GLOBAL' and isinstance(arg, str) and '\n' in arg:
            mod, nm = arg.split('\n', 1)
            parsed.append({'module': mod, 'name': nm})
    if parsed:
        df = pd.DataFrame(parsed).drop_duplicates().reset_index(drop=True)
        st.table(df)
    else:
        st.write("No GLOBAL opcodes found.")

def _safe_preview(pickle_bytes):
    try:
        obj = safe_loads(pickle_bytes)
    except pickle.UnpicklingError as e:
        st.error(f"UnpicklingError: {e}")
        return
    except Exception as e:
        st.error(f"Safe unpickle failed: {e}")
        return

    st.write(f"Type: `{type(obj).__name__}`")
    with st.expander("Object repr / params"):
        try:
            st.text(repr(obj))
        except Exception as repr_err:
            st.warning(f"repr(obj) failed: {repr_err}")
            if hasattr(obj, "get_params"):
                try:
                    st.json(obj.get_params())
                except Exception as p_err:
                    st.error(f"get_params() failed: {p_err}")
            elif hasattr(obj, "__dict__"):
                st.write(pd.Series(obj.__dict__, name="value"))
            else:
                st.write("No fallback representation available.")

    if isinstance(obj, dict):
        st.subheader("Dictionary contents")
        st.dataframe(pd.DataFrame.from_dict(obj, orient='index', columns=['value']), use_container_width=True)
    elif isinstance(obj, (list, tuple, set)):
        st.subheader("Sequence contents")
        st.dataframe(pd.DataFrame({'item': list(obj)}), use_container_width=True)

def _opcode_stats(pickle_bytes):
    names  = [op.name for op, _, _ in pickletools.genops(pickle_bytes)]
    counts = pd.Series(names).value_counts()
    st.bar_chart(counts)
    st.table(counts.rename_axis('opcode').reset_index(name='count'))

def _protocol_versions(pickle_bytes):
    protos = [arg for op, arg, _ in pickletools.genops(pickle_bytes) if op.name=='PROTO']
    if protos:
        df = pd.DataFrame({'protocol': protos})
        st.table(df['protocol'].value_counts().rename_axis('protocol').reset_index(name='count'))
    else:
        st.write("No PROTO opcodes found.")

def _find_large_objects(pickle_bytes):
    large = []
    for op, arg, _ in pickletools.genops(pickle_bytes):
        if op.name in ('BINUNICODE','SHORT_BINUNICODE','BINBYTES','SHORT_BINBYTES'):
            size = len(arg) if hasattr(arg,'__len__') else 0
            if size>1024:
                large.append({'opcode':op.name,'size':size})
    if large:
        st.table(pd.DataFrame(large))
    else:
        st.write("No constants >1KB found.")

def _detect_dangerous(pickle_bytes):
    entries = []
    SAFE_MODULES = {'builtins'}
    for op, arg, _ in pickletools.genops(pickle_bytes):
        if op.name=='GLOBAL' and isinstance(arg,str) and '\n' in arg:
            mod, nm = arg.split('\n',1)
            if mod not in SAFE_MODULES:
                entries.append({'opcode':'GLOBAL','module':mod,'name':nm})
        elif op.name in ('REDUCE','NEWOBJ','STACK_GLOBAL'):
            entries.append({'opcode':op.name,'arg':arg})
    if entries:
        st.table(pd.DataFrame(entries))
    else:
        st.write("No dangerous opcodes detected.")

# Map feature names to handler functions
_FEATURE_HANDLERS = {
    "Metadata":                 _metadata,
    "Disassemble":              _disassemble,
    "List Globals":             _list_globals,
    "Safe Preview":             _safe_preview,
    "Opcode Stats":             _opcode_stats,
    "Protocol Versions":        _protocol_versions,
    "Find Large Objects":       _find_large_objects,
    "Detect Dangerous Objects": _detect_dangerous,
}

def run_feature(feature: str, pickle_bytes: bytes, filename: str):
    feature = feature.strip()
    st.header(f"Pickle Analysis: {feature}")
    st.write(f"Filename: {filename} — {len(pickle_bytes)} bytes")
    handler = _FEATURE_HANDLERS.get(feature)
    if handler:
        handler(pickle_bytes)
    else:
        st.error(f"Unknown feature: {feature}")
