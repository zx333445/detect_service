#!/usr/bin/env python
# coding=utf-8

from routes import app


app.config['JSON_AS_ASCII']=False # 防止json数据中的汉字显示为ascll码
app.config['DEBUG'] = True


if __name__=="__main__":
    app.run(host="192.168.91.63",port="8093") # type: ignore