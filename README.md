## Getting started

Installation.

```bash
cd img2cad
poetry install
```
Run script.
A STEP`file ("output.step") will be generated.

```bash
cd scripts
export OPENAI_API_KEY=<YOUR API KEY>
python cli.py <2D CAD Image File> --output <output file path>
```

Or run streamlit spp

```bash
streamlit run scripts/app.py
streamlit run app.py
```



