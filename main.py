# coding: utf-8

# uvicorn main:app --reload

# from fastapi import FastAPI


# from routers.user import userRouter
# from routers.item import itemRouter

from routers.vnpy import vnpyRouter
# from routers.extend import extendRouter

# from routers.alert import alertRouter
from routers.backtest import backtestRouter
from tortoise.contrib.fastapi import register_tortoise

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, applications
def swagger_monkey_patch(*args, **kwargs):
    return get_swagger_ui_html(
        *args, **kwargs,
        swagger_js_url='https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui-bundle.js',
        swagger_css_url='https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui.css'
    )

applications.get_swagger_ui_html = swagger_monkey_patch

app = FastAPI()

# app.include_router(userRouter,prefix="/user",tags=['user'])
# app.include_router(itemRouter,prefix="/item",tags=['item'])


app.include_router(vnpyRouter,prefix="/vnpy",tags=['vnpy'])
# app.include_router(extendRouter,prefix="/extend",tags=['extend'])

# app.include_router(alertRouter,prefix="/alert",tags=['alert'])
app.include_router(backtestRouter,prefix="/backtest",tags=['backtest'])

register_tortoise(
    app,
    # db_url="sqlite://easy_rbac.sqlite3",
    db_url="mysql://root:yeli123456@@localhost:3306/ldc",
    modules={"models": ["models"]},
    generate_schemas=False,
    add_exception_handlers=True,
)



