import streamlit as st
import requests

API_URL = "http://127.0.0.1:6004"

st.set_page_config(page_title="NLP Service", page_icon="🧠", layout="centered")
st.title("🧠 NLP Text Processing Service")

if "token" not in st.session_state:
    st.session_state.token = None

if "username" not in st.session_state:
    st.session_state.username = None

with st.sidebar:
    st.header("Account")

    if st.session_state.token:
        st.success(f"Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
    else:
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                res = requests.post(
                    f"{API_URL}/login",
                    data={"username": username, "password": password}
                )
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_user = st.text_input("Username", key="reg_user")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Register"):
                res = requests.post(
                    f"{API_URL}/register",
                    json={"username": new_user, "password": new_pass}
                )
                if res.status_code == 200:
                    st.success("Registered! Please log in.")
                else:
                    st.error(res.json().get("detail", "Error"))

if not st.session_state.token:
    st.info("Please log in from the sidebar to use the service.")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}

tab1, tab2, tab3 = st.tabs(["Preprocess", "Sentiment Analysis", "History"])

with tab1:
    st.subheader("Text Preprocessing")
    text = st.text_area("Enter text to preprocess", height=150)
    if st.button("Preprocess"):
        if text.strip():
            res = requests.post(
                f"{API_URL}/preprocess",
                json={"text": text},
                headers=headers
            )
            if res.status_code == 200:
                data = res.json()
                st.markdown("**Cleaned:**")
                st.code(data["cleaned"])
                st.markdown("**Tokens after stopword removal:**")
                st.write(data["tokens_after_stopword_removal"])
                st.markdown("**Final processed:**")
                st.code(data["final_processed"])
            else:
                st.error("Something went wrong. Try logging in again.")
        else:
            st.warning("Please enter some text.")

with tab2:
    st.subheader("Sentiment Analysis")
    text2 = st.text_area("Enter text to analyse", height=150)
    if st.button("Analyse Sentiment"):
        if text2.strip():
            res = requests.post(
                f"{API_URL}/predict",
                json={"text": text2},
                headers=headers
            )
            if res.status_code == 200:
                data = res.json()
                sentiment = data["sentiment"]
                confidence = data["confidence"]
                if sentiment == "positive":
                    st.success(f"Sentiment: POSITIVE — Confidence: {confidence}")
                else:
                    st.error(f"Sentiment: NEGATIVE — Confidence: {confidence}")
                st.markdown("**Processed text:**")
                st.code(data["processed"])
            else:
                st.error("Something went wrong. Try logging in again.")
        else:
            st.warning("Please enter some text.")

with tab3:
    st.subheader("Processing History")
    if st.button("Load History"):
        res = requests.get(f"{API_URL}/history", headers=headers)
        if res.status_code == 200:
            records = res.json()
            if records:
                for r in records:
                    with st.expander(f"#{r['id']} — {r['timestamp']}"):
                        st.markdown(f"**Original:** {r['original']}")
                        st.markdown(f"**Processed:** {r['processed']}")
            else:
                st.info("No history yet.")
        else:
            st.error("Could not load history.")