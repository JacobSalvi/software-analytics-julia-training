import os
import re
from textwrap import dedent
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from src.utils.util import get_model_path
from src.LLM.LLM_predict import predict_llm

def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("julia_logo.png"), style={"float": "right", "height": 60}
    )
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(logo, md=4)])


def textbox(text, box="AI", name="Philippe", app=None):
    """
    Formats user and assistant messages for display in the chatbot.
    """
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "10px 15px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0
        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        thumbnail = html.Img(
            src=app.get_asset_url("julia_logo.png"),
            style={
                "border-radius": "50%",
                "height": "36px",
                "margin-right": "5px",
                "float": "left",
            },
        )

        # Render the text as Markdown with `dcc.Markdown`
        textbox_card = dbc.Card(
            dcc.Markdown(text),  # Render the assistant's response as Markdown
            style=style,
            body=True,
            color="light",
            inverse=False,
        )

        return html.Div([thumbnail, textbox_card])

    else:
        raise ValueError("Incorrect option for `box`.")


AVAILABLE_MODELS = [
    "360m",
    "135m",
    "1.7b",
]

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define Layout
conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

# Add the dropdown to the controls
controls = dbc.InputGroup(
    [
        dbc.Input(id="user-input", placeholder="Write to the chatbot...", type="text"),
        dbc.Select(
            id="model-select",
            options=[{"label": model, "value": model} for model in AVAILABLE_MODELS],
            value="360m",  # Default selected model
            style={"maxWidth": "200px", "marginRight": "10px"},
        ),
        dbc.Button("Submit", id="submit", n_clicks=0, color="primary"),
    ]
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        Header("Julia Chatbot", app),
        html.Hr(),
        dcc.Store(id="store-conversation", data=[]),
        conversation,
        controls,
        dbc.Spinner(html.Div(id="loading-component")),
    ],
)


@app.callback(
    Output("display-conversation", "children"),
    [Input("store-conversation", "data")]
)
def update_display(chat_history):
    """
    Updates the conversation display with the latest chat history.
    """
    messages = []
    for message in chat_history:
        if message['role'] == 'assistant':
            messages.append(textbox(message['content'], box="AI", app=app))
        elif message['role'] == 'user':
            messages.append(textbox(message['content'], box="user"))
    return messages


@app.callback(
    Output("user-input", "value"),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input(n_clicks, n_submit):
    """
    Clears the user input box after submission.
    """
    return ""


def format_response(text):
    """
    Format the assistant response for Markdown rendering.
    - Wraps the text in a Julia code block for proper syntax highlighting.
    """
    return f"```julia\n{text}\n```"


@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data"), State("model-select", "value")],
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history, selected_model):
    """
    Processes the user input and generates a response using the selected model.
    """
    if n_clicks == 0 and n_submit is None:
        return chat_history, None

    if user_input is None or user_input.strip() == "":
        return chat_history, None

    if not chat_history:
        chat_history = []

    # Add the user input to the chat history
    chat_history.append({"role": "user", "content": user_input})

    model_path = get_model_path(selected_model)

    model_type = selected_model.split('_')[0].lower()
    baseline = False
    signature = False
    if "_baseline" in selected_model:
        baseline = True
    elif "_signature" in selected_model:
        signature = True

    try:
        # Generate the assistant reply
        assistant_reply = predict_llm(model_type, user_input, signature, baseline)
        assistant_reply = format_response(assistant_reply)  # Wrap in Markdown code block
        chat_history.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        print(e)
        assistant_reply = "I'm sorry, but I'm unable to assist with that request."
        chat_history.append({"role": "assistant", "content": assistant_reply})

    return chat_history, None


if __name__ == "__main__":
    app.run_server(debug=False)
