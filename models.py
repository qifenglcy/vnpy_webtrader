from enum import IntEnum
from tkinter import FIRST
import types

from tortoise import models, fields


class BaseModel(models.Model):
    id = fields.IntField(pk=True)
    created = fields.DatetimeField(auto_now_add=True)

    class Meta:
        abstract = True

    class PydanticMeta:
        backward_relations = False
        max_recursion = 1


class State(IntEnum):
    # 禁用
    DISABLE = 0
    # 有效
    VALID = 1
    # 删除
    DELETE = 9
    # 激活
    ACTIVE = 5

class LevelType(IntEnum):
    # 一级
    ONE = 0
    # 二级
    TWO = 1
    # 三级
    THREE = 2
   


class MenuType(IntEnum):
    # 按钮
    BUTTON = 3
    # 组件
    ASSEMBLY = 2
    # 目录
    CATALOGUE = 1


class Mixin:
    state = fields.IntEnumField(State, default=State.VALID, description='状态')
    modified = fields.DatetimeField(auto_now=True)


class User(BaseModel, Mixin):
    username = fields.CharField(max_length=20, description='用户名', unique=True, null=False)
    password = fields.CharField(max_length=128, description='密码', null=False)
    mobile = fields.CharField(max_length=128, description='手机', null=False)
    email = fields.CharField(max_length=128, description='邮箱', null=False)
    phone = fields.CharField(max_length=128, description='备用手机', null=True)
    roles: fields.ManyToManyRelation["Role"] = fields.ManyToManyField(
        "models.Role", related_name="users", through="user_role"
    )
    kpis: fields.ManyToManyRelation["Kpi"] = fields.ManyToManyField(
        "models.Kpi", related_name="users", through="user_kpi"
    )

    


class Role(BaseModel, Mixin):
    name = fields.CharField(max_length=20, description='角色名称')
    remark = fields.CharField(max_length=100, description='角色描述', null=True)
    users: fields.ManyToManyRelation[User]
    menus: fields.ManyToManyRelation["Menu"] = fields.ManyToManyField(
        "models.Menu", related_name="roles", through="role_menu"
    )


class Menu(BaseModel, Mixin):
    name = fields.CharField(max_length=20, description='名称', null=True)
    meta = fields.JSONField(description='元数据信息', null=True)
    path = fields.CharField(max_length=128, description='菜单url', null=True)
    type = fields.IntEnumField(MenuType, description='菜单类型', null=True)
    component = fields.CharField( max_length=128, description='组件地址从views目录开始', null=True)
    # parent = fields.ForeignKeyField('models.Menu', on_delete=fields.SET_NULL, related_name='children', null=True)
    parent_id = fields.CharField(max_length=128, description='父级限ID', null=True)
    level = fields.IntEnumField(LevelType, description='菜单等级', null=True)
    roles: fields.ManyToManyRelation[Role]


class Dict(BaseModel, Mixin):
    name = fields.CharField(max_length=10, description='字典名称')
    remark = fields.CharField(max_length=60, description='描述', null=True)
    value = fields.JSONField(description='字典值')

class Auction(models.Model):
    id = fields.IntField(pk=True)
    time = fields.DatetimeField()
    name = fields.CharField(max_length=10, description='品种名称')
    code = fields.CharField(max_length=10, description='品种代码')
    value = fields.FloatField(max_length=80)
    up_down_value = fields.FloatField(max_length=80)
    up_down_percent = fields.FloatField(max_length=80)
    volume = fields.IntField()
    amount = fields.FloatField(max_length=80)

class Symbol(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=10, description='品种名称')
    code = fields.CharField(max_length=10, description='品种代码')
    type = fields.CharField(max_length=10, description='品种类型')
    
class Kpi(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=10, description='KPI名称')
    desc = fields.CharField(max_length=128, description='指标描述')
    weight = fields.FloatField(max_length=20, description='权重')
    type = fields.IntField(max_length=10, description='考核类型')
    
class Dbtradedata(models.Model):
    id = fields.IntField(pk=True)
    gateway_name = fields.CharField(max_length=10, description='接口名称')
    strategy_name = fields.CharField(max_length=10, description='策略名称')
    symbol = fields.CharField(max_length=10, description='品种名称')
    exchange = fields.CharField(max_length=10, description='交易所')
    orderid = fields.IntField()
    tradeid = fields.IntField()
    direction = fields.CharField(max_length=10, description='方向')
    offset = fields.CharField(max_length=10, description='开平')
    price = fields.FloatField(max_length=20, description='权重')
    volume = fields.IntField(description='状态')
    datetime = fields.DatetimeField()

       

