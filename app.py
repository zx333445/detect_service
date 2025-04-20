#!/usr/bin/env python
# coding=utf-8

import io
import torch
import base64
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS,cross_origin

from POS_model import predict

app = Flask(__name__)
CORS(app)  # 解决跨域问题


@app.route("/predict", methods=["GET","POST"])
@cross_origin()
@torch.no_grad()
def pred():
    image = request.files.get("file")
    img_bytes = image.read() # type: ignore
    img = predict(img_bytes)

    f = io.BytesIO()
    img.save(f, 'jpeg')
    #从内存中取出bytes类型的图片
    data = f.getvalue()
    #将bytes转成base64
    data = base64.b64encode(data).decode()
    return data


@app.route("/", methods=["GET", "POST"])
@cross_origin()
def root():
    return render_template("up.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
