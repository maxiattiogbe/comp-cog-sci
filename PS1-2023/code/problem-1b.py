import math
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

coin_flip_sequences = ["HHTHT",
                       "THTTT",
                       "HHHHH",
                       "THTTHTHTHT",
                       "HHTHHHHHTH",
                       "TTTTTTTTTT",
                       "THTTHTTHHTHTHTTHTHTTTHTTHT",
                       "HHTHHHHTHHHTHHHTHHHHHTHHHH",
                       "HHHHHHHHHHHHHHHHHHHHHHHHHH"]

human_data_1 = [[1, 1, 2, 1, 2, 6, 1, 4, 7], 
                [2, 5, 6, 4, 5, 6, 4, 5, 3]]

human_data_2 = [[1, 1, 1, 1, 1, 4, 1, 1, 7],
                [2, 3, 4, 3, 4, 5, 2, 5, 6]]

model_predictions = []

for seq in coin_flip_sequences:
    pd_h1 = (1/2)**(len(seq))

    pd_h2 = 0

    for n in range(0, 101):
        heads_cnt = seq.count("H")
        tails_cnt = seq.count("T")
        θn = n/100

        pd_θn = (θn)**(heads_cnt) * (1 - θn)**(tails_cnt)

        pθn_h2 = 0.01

        pd_h2 += pd_θn * pθn_h2

    x = math.log(pd_h1/pd_h2)
    a = 0.6
    b = 0
    f_x = 1/(1 + math.exp(-a*x + b))

    model_predictions.append(6 * f_x + 1)

model_predictions = [7 - num for num in model_predictions]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Cover Story 1: Strange Coins on Sidewalk", "Cover Story 2: Fresh Coins from Bank"))

fig.add_trace(
    go.Scatter(
        x=model_predictions,
        y=human_data_1[0],
        mode="markers",
        name="Me"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=model_predictions,
        y=human_data_1[1],
        mode="markers",
        name="Majd Alafrange"
    ),
    row=1, col=1
)


fig.add_trace(
    go.Scatter(
        x=model_predictions,
        y=human_data_2[0],
        mode="markers",
        name="Me"
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=model_predictions,
        y=human_data_2[1],
        mode="markers",
        name="Andi Qu"
    ),
    row=1, col=2
)

fig.update_xaxes(title_text = "Model Predictions", row = 1, col = 1)
fig.update_xaxes(title_text = "Model Predictions", row = 1, col = 2)

fig.update_yaxes(title_text = "Human Data", row = 1, col = 1)
fig.update_yaxes(title_text = "Human Data", row = 1, col = 2)

fig.update_layout(height=600, width=1000)

fig.show()

model_predictions += model_predictions
human_data_1 = human_data_1[0] + human_data_1[1]
human_data_2 = human_data_2[0] + human_data_2[1]

corrcoeff1 = np.corrcoef(model_predictions, human_data_1)[0, 1]
corrcoeff2 = np.corrcoef(model_predictions, human_data_2)[0, 1]

print(corrcoeff1)
print(corrcoeff2)
print()