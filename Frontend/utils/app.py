from flask import Flask, render_template, redirect, url_for
import gradio as gr

app = Flask(__name__)

@app.route('/')
def test1():
    return render_template('test1.html')
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page1/try1')
def try1():
    return render_template('try1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)

