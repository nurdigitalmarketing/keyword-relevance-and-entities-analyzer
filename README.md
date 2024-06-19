![Keyword Relevance & Sentiment Analysis Tool](https://raw.githubusercontent.com/nurdigitalmarketing/keyword-relevance-and-sentiment-app/main/keyword-relevance-and-sentiment-app.png)

# Keyword Relevance & Sentiment Analysis Tool

This Streamlit application is designed to provide intuitive and detailed text analysis focused on keyword extraction, keyword relevance, and sentiment analysis. It uses the power of advanced Python libraries such as `spaCy`, `PyTextRank`, and `TextBlob` to process and analyze input text.

## 👉🏼 Main Features

- **Keyword Extraction**: Identifies and evaluates the most relevant keywords within the input text, using the PyTextRank algorithm.
- **Sentiment Analysis**: Provides sentiment analysis of the text, evaluating both polarity (positive, neutral, negative) and subjectivity (objective vs. subjective).
- **Target Keywords**: Allows users to enter specific target keywords and assesses their relevance in the context of the analyzed text.

## 👉🏼 How to Use

To run this application on your local system, follow these steps:

1. Make sure you have Python installed on your system.
2. Clone this repository or download the source files.
3. Install the necessary dependencies using `pip`:
    ```
    pip install -r requirements.txt
    ```
4. Start the Streamlit application:
    ```
    streamlit run streamlit_app.py
    ```

## 👉🏼 Technologies Used

- `streamlit`
- `spacy`
- `pytextrank`
- `textblob`
- `pandas`

## 👉🏼 Example Usage

After starting the application, enter the text to be analyzed in the text area provided and specify any target keywords separated by commas. Press the "Analyze" button to view the results.

## 👉🏼 License

This project is currently without a specific license. All rights are reserved by the author. For further information or requests for use, please [contact the author](https://www.nur.it).

## 👉🏼 Author

This project was developed by [NUR® Digital Marketing](https://www.nur.it/), inspired by the guide available on Analytics Vidhya: [Keyword Extraction Methods from Documents in NLP](https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp).
