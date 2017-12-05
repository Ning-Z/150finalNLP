from flask import Flask, request, redirect, render_template, url_for
app = Flask(__name__)

from complifier import *

@app.route("/")
def index():
	createlink = "<a href='" + url_for('create') + "'>start to complify</a>"
	return """<html><head><title>home</title></head><body>"""+createlink+"""</body></html>"""

@app.route('/create',methods=['GET', 'POST'])
def create():
	if request.method == 'GET':
		return render_template("index.html")
	elif request.method == 'POST':
		input = request.form['input'];
		output = complify(input)
		return render_template("answer.html",input=input, output=output)
	else:
		return "<h2>Invalid input</h2>"

if __name__ == "__main__":
    app.run(debug=True)
