#!/usr/bin/env python3
from bottle import route, run, post, request, static_file

@route('/')
def server_static(filepath="demo.html"):
    return static_file(filepath, root='./public/')

@post('/doform')
def process():

    name = request.forms.get('name')
    occupation = request.forms.get('occupation')
    return "Your name is {0} and you are a(n) {1}".format(name, occupation)

run(host='localhost', reloader=True, port=8080, debug=True)