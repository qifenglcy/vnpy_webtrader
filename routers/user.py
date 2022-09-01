from calendar import week
from re import X
from time import time
from typing import Optional
from tortoise import Tortoise
import pandas as pd

from crud import BaseCrud
from models import User, Role, Menu, Dict, Auction,Symbol,Kpi,Dbtradedata

from schema import (
    UserLogin,
    Response,
    Token,
    UserOut,
    UserIn,
    UserKIn,
    RoleIn,
    RoleOut,
    MenuIn,
    DictIn,
    DictOut,
    MenuOut,
    UserQuery,
    RoleQuery,
    Userstate,
    Userroles,
    MenuList,
    AuctionOut,
    AuctionIn,
    AuctionQuery,
    SymbolOut,
    SymbolIn,
    KpiOut,


)

from utils import (
    verify_password,
    create_access_token,
    get_password_hash,
    list_to_tree,
    auth_token,
    api_premiss,
)
from fastapi import APIRouter, Depends




userRouter=APIRouter()
app=userRouter

# @app.post("/login", summary="登录", response_model=Response[Optional[Token]])
# async def user_auth(form: UserLogin):
#     user_obj = await User.get(username=form.username, state=1)
#     if user_obj :
#         if verify_password(form.password, user_obj.password):
#             token = create_access_token({"sub": user_obj.username})
#             return Response(data=Token(id=user_obj.id, access_token=token, token_type="bearer" ))
#         return Response(code=1, msg="账号或密码错误")
#     return Response(code=1, msg="账号被禁用")


@app.get("/user", summary="用户列表", response_model=Response)
async def user_list():
    result = User.filter(state__not=9).all()
    return Response(data=await UserOut.from_queryset(result))

@app.post("/user/query", summary="模糊查询")
async def user_query(form: UserQuery):
    query_dict = {}
    for k, v in form.dict().items():
        if "id" not in k and v is not None :
            query_dict[f"{k}__contains"] = v
    result = User.filter(**query_dict, state__not=9)
    return Response(data=await UserOut.from_queryset(result))


@app.post("/user_add", summary="用户新增", response_model=Response[Optional[UserOut]])
async def user_add(form: UserIn):
    if await User.filter(username=form.username).first():
        return Response(code=1, msg="用户名已存在")
    form.password = get_password_hash(form.password)
    role_objs = [await Role.get(pk=rid) for rid in form.roles]
    del form.roles
    user_obj = await User.create(**form.dict())
    await user_obj.roles.add(*role_objs)
    return Response(data=await UserOut.from_tortoise_orm(user_obj))


@app.get("/user_info/{pk}", summary="用户信息", response_model=Response)
async def user_info(pk: int):
    user_obj = await User.get(id=pk)
    return Response(data=await UserOut.from_tortoise_orm(user_obj))



@app.delete("/user_cancel/{pk}", summary="删除用户", response_model=Response)
async def user_del(pk: int):
    # todo 1. 用户不存在无法删除, 2. 用户已删除状态 无法删除
    if await User.filter(pk=pk).update(state=9):
        return Response()
    return Response(msg="用户不存在")


@app.put("/user_update/{pk}", summary="更新用户信息", response_model=Response)
async def user_put(pk: int, form: UserIn):
    # todo 删除的角色不能选
    user_obj = await User.get(id=pk)
    role_objs = [await Role.get(pk=rid) for rid in form.roles]
    del form.roles
    await user_obj.roles.clear()
    await user_obj.roles.add(*role_objs)
    await User.filter(id=pk).update(**form.dict())
    return Response(data=await UserOut.from_tortoise_orm(user_obj))

@app.put("/userstate_update/{pk}", summary="更新用户状态", response_model=Response)
async def user_put(pk: int, form: Userstate):
    # todo 删除的角色不能选
    user_obj = await User.get(id=pk)
    await User.filter(id=pk).update(**form.dict())
    return Response(data=await UserOut.from_tortoise_orm(user_obj))



@app.get("/role", summary="角色列表", response_model=Response)
async def role_list():
    return Response(data=await RoleOut.from_queryset(BaseCrud.page_query(Role)))
   

@app.get("/role_info/{pk}", summary="角色信息", response_model=Response)
async def role_info(pk: int):
    role_obj = await Role.get(id=pk)
    return Response(data=await RoleOut.from_tortoise_orm(role_obj))


@app.get("/role_menu/{pk}", summary="角色权限", response_model=Response)
async def role_info(pk: int):
    conn = Tortoise.get_connection("default")
    # info = await conn.execute_query_dict(
    info = await conn.execute_query_dict(
        "select  name, remark,state FROM role WHERE id = (?)", [pk]
    )
    #print('info',info)
    menus = await conn.execute_query_dict(
        "select m.id, m.name,m.meta,m.path,m.parent_id,mr.status FROM menu as m, role_menu as mr WHERE m.id = mr.menu_id AND role_id = (?)AND mr.status = 1",
        [pk],
    )
    permissions = await conn.execute_query_dict(
        """
        select m.prem_tag from role_menu as rm LEFT JOIN menu as m ON rm.menu_id = m.id WHERE rm.role_id = (
        SELECT ur.role_id FROM user_role as ur WHERE ur.user_id = (?) AND ur.status = 0
        ) AND m.type = 2 AND m.prem_tag NOTNULL
        """,
        [pk],
    )
    if info:
        info[0].update({"menus": menus})
       # info[0].update(
           # {"permissions": [permission["prem_tag"] for permission in permissions]}
        #)
        return Response(data=info[0])
    return Response(msg="角色不存在")


@app.post("/role_add", summary="角色新增")
async def role_list(form: RoleIn):
    role_obj = await BaseCrud.m2m_create(
        Role, form, m2=Menu, del_field="menus", m2_field="menus"
    )
    return Response(data=await RoleOut.from_tortoise_orm(role_obj))


@app.put("/role_update/{pk}", summary="角色更新")
async def role_put(pk: int, form: RoleIn):
    #await BaseCrud.m2m_update(pk, Role, form, Menu, del_field="menus", m2_field="menus")
    # todo 删除的角色不能选
    role_obj = await Role.get(id=pk)
    menu_objs = [await Menu.get(pk=rid) for rid in form.menus]
    del form.menus
    await role_obj.menus.clear()
    await role_obj.menus.add(*menu_objs)
    await Role.filter(id=pk).update(**form.dict())
    return Response(data=await RoleOut.from_tortoise_orm(role_obj))


@app.delete("/role_cancel/{pk}", summary="角色删除", response_model=Response)
async def role_delete(pk: int):
   # todo 1. 角色不存在无法删除, 2. 角色已删除状态 无法删除
    if await Role.filter(pk=pk).update(state=9):
        return Response()
    return Response(msg="角色不存在")

    

@app.post("/role/query", summary="检索角色列表")
async def role_query(form: RoleQuery):
    skip = (form.page - 1) * form.limit
    limit = form.limit
    delattr(form, "page")
    delattr(form, "limit")
    result = Role.filter(**form.dict(), state__not=9).all().offset(skip).limit(limit)
    return Response(data=await RoleOut.from_queryset(result))


@app.get("/menu", summary="菜单列表")
async def menu_list():
    result = BaseCrud.page_query(Menu)
    return Response(data=await MenuOut.from_queryset(result))


@app.post("/menu", summary="菜单新增")
async def menu_add(form: MenuIn):
    return Response(
        data=await MenuOut.from_tortoise_orm(await Menu.create(**form.dict()))
    )


@app.post("/dict", summary="字典新增")
async def dict_add(form: DictIn):
    return Response(
        data=await DictOut.from_tortoise_orm(await Dict.create(**form.dict()))
    )
@app.get("/role/menu", summary="查询全部菜单树")
async def common_menu_tree():
    data = await Menu.all().order_by("-state").values()
    return Response(data=list_to_tree(data))


@app.get("/role/{pk}/menu", summary="查询角色菜单树")
async def common_menu_tree(pk: int):
    data = await Menu.filter(roles__id=pk).all().order_by("-state").values()  
    return Response(data=list_to_tree(data))


@app.put("/role/{pk}/active", summary="激活角色")
async def current_role(pk: int, obj: User = Depends(auth_token)):
    """
    select ur.user_id, ur.role_id, u.username, r.name, r.remark, ur.status FROM user as u, role as r, user_role as ur
    where u.id = ur.user_id AND ur.role_id = r.id
    """
    conn = Tortoise.get_connection("default")
    # 查出用户激活的角色，改为未激活
    sql = "select ur_id FROM user_role WHERE user_id = (?) AND status = 0"
    for ur_id in await conn.execute_query_dict(sql, [obj.id]):
        await conn.execute_query(
            "update user_role set status = 1 where ur_id = (?)", [ur_id.get("ur_id")]
        )

    await conn.execute_query(
        "update user_role set status = 0 WHERE user_id = (?) AND role_id = (?)",
        [obj.id, pk],
    )
    return Response(data=await RoleOut.from_queryset_single(Role.get(id=pk)))



@app.get("/settings", summary="系统设置", response_model=Response[DictOut])
async def get_settings():
    return Response(data=await DictOut.from_queryset_single(Dict.get(name="系统设置")))


@app.get("/auction", summary="股票竞价列表")
async def auction_list():
    db = Tortoise.get_connection("default")
    info = await db.execute_query_dict("SELECT a.code,s.code,a.time,a.name,a.volume,a.amount,s.type FROM auction as a,symbol as s WHERE a.code = s.code  AND  TO_DAYS(a.time) = TO_DAYS(NOW())")
    return Response(data=info)
    

@app.get("/auction/{code}", summary="股票竞价日数据")
async def auction_list(code: str):
   db = Tortoise.get_connection("default")
   result = await db.execute_query_dict("SELECT a.code,s.code,a.time,a.name,a.volume,a.amount,s.type FROM auction as a,symbol as s WHERE a.code = s.code  AND a.code = %s", [code])
   
   return Response(data=result)


@app.get("/auction/week/{code}", summary="股票竞价周数据")
async def auction_list(code: str):
   db = Tortoise.get_connection("default")
   result = await db.execute_query_dict("SELECT * FROM auction WHERE code = %s", [code])
   results = pd.DataFrame(result)
   results.index = results['time']
   results['Week']=results.index.isocalendar().week
   data1 = results.groupby(results.index.isocalendar().week).min()[['time','Week']]
   data3 = results.groupby(results.index.isocalendar().week).sum()[['amount','volume']]
   data3['Week'] = data3.index
   data = pd.merge(data1, data3, on='Week')
   week_data=[]
   for contract,row in data.iterrows():
       week_data.append([row['time'],row['amount'],row['volume']])
   
   return Response(data=week_data)


@app.get("/auction/month/{code}", summary="股票竞价月数据")
async def auction_list(code: str):
   db = Tortoise.get_connection("default")
   result = await db.execute_query_dict("SELECT * FROM auction WHERE code = %s", [code])
   results = pd.DataFrame(result)
   results.index = results['time']
   results['month']=results.index.month
   data1 = results.groupby(results.index.month).min()[['time','month']]
   data3 = results.groupby(results.index.month).sum()[['amount','volume']]
   data3['month'] = data3.index
   data = pd.merge(data1, data3, on='month')
   month_data=[]
   for contract,row in data.iterrows():
       month_data.append([row['time'],row['amount'],row['volume']])
   
   return Response(data=month_data)

@app.get("/auction/year/{code}", summary="股票竞价年数据")
async def auction_list(code: str):
   db = Tortoise.get_connection("default")
   result = await db.execute_query_dict("SELECT * FROM auction WHERE code = %s", [code])
   results = pd.DataFrame(result)
   results.index = results['time']
   results['year']=results.index.month
   data1 = results.groupby(results.index.month).min()[['time','year']]
   data3 = results.groupby(results.index.month).sum()[['amount','volume']]
   data3['year'] = data3.index
   data = pd.merge(data1, data3, on='year')
   year_data=[]
   for contract,row in data.iterrows():
       year_data.append([row['time'],row['amount'],row['volume']])
   
   return Response(data=year_data)


@app.get("/symbol", summary="品种列表")
async def symbol_list():
   db = Tortoise.get_connection("default")
   result = await db.execute_query_dict("SELECT * FROM symbol")
   return Response(data=result)


@app.post("/symbol_add", summary="品种新增", response_model=Response[Optional[SymbolOut]])
async def user_add(form: SymbolIn):
    if await Symbol.filter(name=form.name).first():
        return Response(code=1, msg="品种已存在")
    symbol_obj = await Symbol.create(**form.dict())
    return Response(data=await SymbolOut.from_tortoise_orm(symbol_obj))

@app.get("/KPI", summary="KPI指标列表")
async def kpi_list():
    result = Kpi.filter(type=1).all()
    
    return Response(data=await KpiOut.from_queryset(result))

@app.get("/show", summary="综合表现列表")
async def kpi_list():
    result = Kpi.filter(type=2).all()
    
    return Response(data=await KpiOut.from_queryset(result))

@app.get("/user/{pk}/kpi", summary="查询用户KPI指标")
async def common_kpi(pk: int):
    db = Tortoise.get_connection("default")
    kpis = await db.execute_query_dict(
        "select k.id,k.name,k.desc,k.weight,k.type,uk.first_score, uk.review_score,uk.month,uk.first_desc,uk.review_desc FROM kpi as k, user_kpi as uk WHERE k.id = uk.kpi_id AND k.type = 1 AND user_id = %s ", [pk])
    print(kpis)
    return Response(data=kpis)

@app.get("/user/{pk}/show", summary="查询用户综合表现指标")
async def common_show(pk: int):
    db = Tortoise.get_connection("default")
    kpis = await db.execute_query_dict(
         "select k.id,k.name,k.desc,k.weight,k.type,uk.first_score, uk.review_score,uk.month,uk.first_desc,uk.review_desc FROM kpi as k, user_kpi as uk WHERE k.id = uk.kpi_id AND k.type = 2 AND user_id = %s ", [pk])
    return Response(data=kpis)

@app.post("/userKpi_add", summary="用户考核新增")
async def userKpi_list(form: UserKIn):
    userKpi_obj = await BaseCrud.m2m_create(
        User, form, m2=Kpi, del_field="kpis", m2_field="kpis"
    )
    return Response(data=await UserOut.from_tortoise_orm(userKpi_obj))



@app.get("/dbtradedata", summary="成交记录列表")
async def dbtradedata_list():
   db = Tortoise.get_connection("default")
   result = await db.execute_query_dict("SELECT * FROM dbtradedata")
   return Response(data=result)

@app.get("/dbtradedata/{code}", summary="单个策略成交记录")
async def dbtradedata_list(code: str):
   db = Tortoise.get_connection("default")
   result = await db.execute_query_dict("SELECT * FROM dbtradedata WHERE strategy_name = %s", [code])
   
   return Response(data=result)


