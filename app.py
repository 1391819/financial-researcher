# Import baseline dependencies
import time
from datetime import date

import pandas as pd
import pandas_datareader as data
import requests
import streamlit as st
from bs4 import BeautifulSoup
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
# summarisation (Pegasus) and sentiment analysis (BERT) models
from transformers import (BertForSequenceClassification, BertTokenizer,
                          PegasusTokenizer, TFPegasusForConditionalGeneration,
                          pipeline)

# Setting streamlit page config to wide
st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True, show_spinner=False)
# Setup summarisation model
def get_summarisation_model():
    sum_model_name = "human-centered-summarization/financial-summarization-pegasus"
    sum_tokenizer = PegasusTokenizer.from_pretrained(sum_model_name)
    sum_model = TFPegasusForConditionalGeneration.from_pretrained(
        sum_model_name)

    # returning model and tokenizer
    return sum_model, sum_tokenizer


@st.cache(allow_output_mutation=True, show_spinner=False)
# Setup sentiment analysis model
def get_sentiment_pepeline():
    sen_model_name = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
    sen_tokenizer = BertTokenizer.from_pretrained(sen_model_name)
    sen_model = BertForSequenceClassification.from_pretrained(
        sen_model_name, num_labels=3)
    sentiment_nlp = pipeline("sentiment-analysis",
                             model=sen_model, tokenizer=sen_tokenizer)

    # returning sentiment pipeline
    return sentiment_nlp


@st.cache(show_spinner=False, suppress_st_warning=True)
# Get all links from Google News
def search_urls(ticker, num, date):

    # https://developers.google.com/custom-search/docs/xml_results_appendices#interfaceLanguages

    # Request headers and parameters
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
    }

    params = {
        "as_sitesearch": "finance.yahoo.com",  # we only want results from Yahoo Finance
        "hl": "en",  # language of the interface
        "gl": "us",  # country of the search
        "tbm": "nws",  # news results
        "lr": "lang_en"  # language filter
    }

    # base URL
    url = "https://www.google.com/search"

    # search query
    params["as_epq"] = ticker
    params["as_occt"] = ticker
    # number of search results per page
    params["num"] = num

    # articles timeframe
    #  d = past 24h, h = past hour, w = past week, m = pasth month
    if date == "Past week":
        params["as_qdr"] = "w"
    elif date == "Past day":
        params["as_qdr"] = "d"

    r = requests.get(url, headers=headers, params=params,
                     cookies={'CONSENT': 'YES+'})
    time.sleep(5)
    st.write("Searched URL:")
    st.write(r.url)  # debugging
    soup = BeautifulSoup(r.text, "html.parser")
    atags = soup.find_all("a", "WlydOe")
    hrefs = [link["href"] for link in atags]

    return hrefs


@st.cache(show_spinner=False)
# Extract title, date, and content of the article from all given URLs
def search_scrape(urls):
    articles = []
    titles = []
    post_dates = []

    for url in urls:
        r = requests.get(url)
        time.sleep(5)
        soup = BeautifulSoup(r.text, "html.parser")

        # title
        title = soup.find("header", "caas-title-wrapper")
        # handling missing titles
        if title is not None:
            titles.append(title.text)
        else:
            titles.append("N/A")

        # posting date of the article
        date = soup.find("time", "caas-attr-meta-time")
        # handling missing dates
        if date is not None:
            post_dates.append(date.text)
        else:
            post_dates.append("N/A")

        # article content
        # all the paragraphs within the article
        paragraphs = soup.find_all("div", "caas-body")
        text = [paragraph.text for paragraph in paragraphs]
        # extract only the first 300 words (needs to be done to avoid limit
        # problems with the summarisation model)
        words = " ".join(text).split(" ")[:350]
        article = " ".join(words)
        articles.append(article)

    return titles, post_dates, articles


@st.cache(show_spinner=False)
# Summarise all given articles using a fine-tuned Pegasus Transformers model
def summarise_articles(sum_model, sum_tokenizer, articles):
    summaries = []
    for article in articles:

        # source
        # https://huggingface.co/human-centered-summarization/financial-summarization-pegasus
        input_ids = sum_tokenizer(
            article, return_tensors="tf").input_ids
        output = sum_model.generate(
            input_ids, max_length=55, num_beans=5, early_stopping=True)
        summary = sum_tokenizer.decode(
            output[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries


@st.cache(show_spinner=False)
# Join all data into rows
def create_output_array(titles, post_dates, summarised_articles, sentiment_scores, raw_urls):
    output_array = []
    for idx in range(len(summarised_articles)):
        row = [
            titles[idx],
            post_dates[idx],
            summarised_articles[idx],
            sentiment_scores[idx]["label"].capitalize(),
            "{:.0%}".format(sentiment_scores[idx]["score"]),
            raw_urls[idx]
        ]
        output_array.append(row)

    return output_array


@st.cache(show_spinner=False)
# Convert dataframe to .csv file
def convert_df(df):
    return df.to_csv().encode("utf-8")

# ------------------------------------------------------------------------------


@st.cache(show_spinner=False)
# Load data from Yahoo Finance
def load_data(ticker, start, end):
    df = data.DataReader(ticker, "yahoo", start, end)
    df.reset_index(inplace=True)
    return df


@st.cache(show_spinner=False)
# Predict stock trend for N years using Prophet
def predict(df, period):

    df_train = df[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet()

    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    return model, forecast


def main_page():

    # Financial News Analysis feature

    # Streamlit text

    st.sidebar.markdown("## Financial News Analysis")
    st.sidebar.write(
        "Scrape, auto summarise and calculate sentiment for stock and crypto news.")

    # User input
    ticker = st.text_input("Ticker:", "TSLA")
    num = st.number_input("Number of articles:", 5, 15, 10)
    date = st.selectbox(
        "Timeline:", ["Past week", "Past day"])

    search = st.button("Search")

    st.info("Please do not spam the search button")
    st.markdown("---")

    # If button is pressed
    if search:

        with st.spinner("Processing articles, please wait..."):
            # Search query and return all articles' links
            raw_urls = search_urls(ticker, num, date)

            # If any problems happened (e.g., blocked by Google's server) stop app
            if not raw_urls:
                st.error("Please wait a few minutes before trying again")
            else:

                # Scrap title, posting date and article content from all the URLs
                titles, post_dates, articles = search_scrape(raw_urls)

                # Summarise all articles
                summarised_articles = summarise_articles(
                    sum_model, sum_tokenizer, articles)

                # Calculate sentiment for all articles
                # source
                # https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis
                sentiment_scores = sentiment_pipeline(summarised_articles)

                # Create dataframe
                output_array = create_output_array(
                    titles, post_dates, summarised_articles, sentiment_scores, raw_urls)
                cols = ["Title", "Date", "Summary",
                        "Label", "Confidence", "URL"]
                df = pd.DataFrame(output_array, columns=cols)

                # Visualise dataframe
                st.dataframe(df)

                # Convert dataframe to csv and let user download it
                csv_file = convert_df(df)

                # Download CSV
                st.download_button(
                    "Save data to CSV", csv_file, "assetsummaries.csv", "text/csv", key="download-csv")


def page2():

    # Stock Trend Forecasting feature

    # Streamlit text
    st.sidebar.markdown("## Stock Trend Forecasting")
    st.sidebar.write(
        "A simple dashboard for stock trend forecasting and analysis.")

    # Start and end date of data
    start = "2010-01-01"
    end = date.today().strftime("%Y-%m-%d")

    # Ticker selection
    ticker = st.text_input("Ticker:", "AAPL")
    # Loading data from Yahoo Finance
    df = load_data(ticker, start, end)

    # Period selection
    n_years = st.number_input("Years of prediction:", 1, 4, 1)
    period = n_years * 365

    # Start prediction button
    init = st.button("Predict")

    st.markdown("---")

    # Visualisation
    # Dropping adj close column
    df = df.drop(["Adj Close"], axis=1)

    # Visualisation
    # Exploratory analysis
    st.subheader("Exploratory analysis")
    st.write(df.describe())

    # Plot raw closing data with 100 and 200 days MA (for simple analysis)
    st.subheader("Closing data, MA100 and MA200")

    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()

    fig = go.Figure()
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
            pad=4
        )
    )
    fig.add_trace(go.Scatter(x=df["Date"],
                  y=df['Close'], name="stock_close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=ma100, name="ma100"))
    fig.add_trace(go.Scatter(x=df["Date"], y=ma200, name="ma200"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # If button is pressed, start forecasting
    if init:
        with st.spinner("Please wait..."):
            model, forecast = predict(df, period)

            st.markdown("---")
            st.subheader("Forecast data")
            st.write(forecast.tail())

            st.subheader(f"Forecast plot for {n_years} years")

            fig = plot_plotly(model, forecast)
            fig.update_layout(
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=0,
                    pad=4
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast components")
            fig = model.plot_components(forecast)
            st.write(fig)


if __name__ == "__main__":

    with st.spinner("Loading all models..."):
        # Creating summariser and sentiment models
        sum_model, sum_tokenizer = get_summarisation_model()
        sentiment_pipeline = get_sentiment_pepeline()

    page_names_to_funcs = {
        "Financial News Analysis": main_page,
        "Stock Trend Forecasting": page2
    }

    st.sidebar.markdown("# Financial Researcher")

    selected_page = st.sidebar.selectbox(
        "Select a page", page_names_to_funcs.keys())

    st.sidebar.markdown("---")

    page_names_to_funcs[selected_page]()
