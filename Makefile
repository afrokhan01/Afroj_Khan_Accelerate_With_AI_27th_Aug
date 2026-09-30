install:
	pip install -r requirements.txt

test:
	pytest

run:
	python -m app.main --env dev

ui:
	streamlit run streamlit_app.py

ui-idamp:
	streamlit run app/streamlit_app.py

ui-original:
	streamlit run streamlit_app.py
