import plotly.express as px


def main():
    # https://plotly.com/python/2D-Histogram/
    # https://plotly.com/python-api-reference/generated/plotly.express.density_heatmap.html

    df = px.data.tips()

    # fig = px.density_heatmap(df, x="total_bill", y="tip", text_auto=True)

    # fig = px.density_heatmap(df, x="total_bill", y="tip", nbinsx=20, nbinsy=20, color_continuous_scale="Viridis")

    # fig = px.density_heatmap(df, x="total_bill", y="tip", nbinsx=20, nbinsy=20, color_continuous_scale="Viridis",
    #                          text_auto=True)

    # fig = px.density_heatmap(df, x="total_bill", y="tip", z="size", histfunc="avg", nbinsx=20, nbinsy=20,
    #                          color_continuous_scale="Viridis", text_auto=True)

    fig = px.density_heatmap(df, x="total_bill", y="tip", z="size", histfunc="avg", nbinsx=20, nbinsy=20,
                             text_auto=True)

    fig.show()


if __name__ == '__main__':
    main()
