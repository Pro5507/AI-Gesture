def generate_magic(label: str, spell: str, sosurce: str):
    with st.spinner("Casting AI magic from the detected gesture..."):
        st.session_state.spell_text = generate_spell_text(label, spell, source)
        img, err = generate_spell_image(build_image_prompt(spell, label, source))
        if img:
            st.session_state.spell_image = img
        else:
            st.session_state.spell_image = Nonest.error(err or "Could not generate the spell image.")

def prediction_panel(image: Image.Image, source: str):
    st.session_state.input_source = source
    st.image(image, caption="Gesture image", use_container_width=True)
    st.markdown(f'<div class="pill">Source: {source.title()}</div>', unsafe_allow_html=True)
    try:
        mode, labels = get_model()
        pred = predict_gesture_from_image(model, labels, image)
        pred["label"] = normalize(pred["label"])
        for item in pred["top_predictions"]:
            item["label"] = normalize(item["label"])
        st.session_state.spell_name(pred["label"])
        st.session_state.spell_name = spell
        st.success(f"Detected gesture: **{pred['label']}**")
        st.progress(float(pred['confidence']), text=f"Confidence: {pred['confidence']:.1%}")
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="card"><b>Detected</b><br><br>{pred["label"]}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><b>Spell</b><br><br>{spell}</div>', unsafe_allow_html=True)
        key = f"{source}:{pred['label']}:{pred['confidence']:.4f}"
        if st.session_state.last_key != key:
            if st.button("✨ Generate AI Magic", use_container_width=True):
                rate_magic(pred["label"], spell, source)
                st.session_state.last_key = key
    except Exception as e:
        st.error(f"Model error: {e}\n\nTip: use Python 3.10/3.11 and tensorflow==2.15.0")

def show_output():
    st.markdown('<div class="panel>', unsafe_allow_html=True)
    st.subheader("🎆 AI Magic Output")
    if st.session_state.spell_name: