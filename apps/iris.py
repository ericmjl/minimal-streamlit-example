from sklearn.datasets import load_iris
import pandas as pd
import streamlit as st
import holoviews as hv
import hvplot.pandas  # noqa: F401


st.header("Holoviews!")
st.write("As a bonus, I'm going to show you a holoviews plot in streamlit!")


def iris_data():
    data = load_iris()
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    return df


df = iris_data()

x_column = st.selectbox("X axis", df.columns)
y_column = st.selectbox("Y axis", df.columns)

plot = (
    iris_data().hvplot.scatter(x_column, y_column).opts(width=400, height=400)
)

st.bokeh_chart(hv.render(plot, backend="bokeh"))
