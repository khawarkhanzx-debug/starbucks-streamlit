# Starbucks EDA & Decision Tree

Run: `pip install -r requirements.txt`, then `streamlit run app.py`.

Upload the original Starbucks menu CSV in the sidebar. For a default dataset,
place it at `data/starbucks.csv` before deploying. No CSV or fitted model was
included in the supplied notebook; notebook outputs are not a full dataset.

The application follows the notebook's depth-5 Decision Tree and seed 42.
It excludes beverage names/preparation and fully empty columns, fits median
imputation on training rows only, and builds the form from actual features.
Results may differ from the original notebook because preprocessing leakage
has been corrected. The original fitted model cannot be recovered from the notebook.

Deploy on Streamlit Community Cloud by selecting the GitHub repository,
branch and `app.py` entrypoint (or its subfolder path). Python 3.11 recommended.
The app supports CSV uploads even when no dataset is bundled.

This is an educational project, not an official Starbucks service.
