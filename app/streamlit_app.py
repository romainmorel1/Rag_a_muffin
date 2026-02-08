import streamlit as st
from src.llm_mistral import MistralLLM
from src.prompts import CHEF_MUFFIN_SYSTEM
from src.rag_app import MuffinRAG

st.set_page_config(page_title="Chef Muffin 🧁", page_icon="🧁")
st.title("Chef Muffin 🧁 — RAG 100% Muffins")

rag = MuffinRAG()
llm = MistralLLM(model="mistral-large-latest")

query = st.text_area("Dis-moi ce que tu as (ingrédients, envie, sucré/salé)…",
                     placeholder="Ex: fromage de chèvre, épinards, farine, œufs…")

if st.button("Aide-moi à cuisiner mon muffin"):
    if not query.strip():
        st.warning("Écris au moins un ingrédient ou une envie 🙂")
    else:
        with st.spinner("Chef Muffin réfléchit…"):
            # metas = rag.retrieve(query)
            answer = rag.answer(query=query, llm=llm, system_template=CHEF_MUFFIN_SYSTEM)

        st.markdown("### 🧁 Réponse de Chef Muffin")
        st.markdown(answer)

        # if metas:
        #     st.markdown("---")
        #     st.markdown("### 📚 Recettes candidates (issues de la base)")
        #     for m in metas:
        #         title = m.get("title_fr", "Sans titre")
        #         link = m.get("link", "")
        #         source = m.get("source", "")
        #         if link:
        #             st.markdown(f"- **{title}** — [{source}]({link})")
        #         else:
        #             st.markdown(f"- **{title}** — {source}")

