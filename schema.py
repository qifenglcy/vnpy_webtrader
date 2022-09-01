from calendar import month
from datetime import date
from typing import Optional, List
from typing import TypeVar, Generic
from pydantic import BaseModel
from pydantic.generics import GenericModel
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator

from models import User, Role, Menu, Dict, Auction,Symbol,Kpi,Dbtradedata
DataT = TypeVar('DataT')


class UserLogin(BaseModel):
    username: str
    password: str


class Response(GenericModel, Generic[DataT]):
    code: int = 0
    msg: str = 'ok'
    data: Optional[DataT]


class Token(BaseModel):
    id: int
    access_token: str
    token_type:str


#class BasicQuery(BaseModel):
    # limit: int = 10
    # page: int = 1


Tortoise.init_models(["models"], "models")

UserOut = pydantic_model_creator(User, name="User", exclude=())
UserInBasic = pydantic_model_creator(User, name="UserIn", exclude_readonly=True)


class UserIn(UserInBasic):
    roles:List[int]

class UserKIn(BaseModel):
    kpis:List[int]


#class UserQuery(BasicQuery):
    #username: str = None

class UserQuery(BaseModel):
    username: str = None

class Userstate(BaseModel):
    state: int = 1

class Userroles(BaseModel):
     roles:List[int]


#RoleOut = pydantic_model_creator(Role, name="Role", exclude=("users", "menus"))
RoleOut = pydantic_model_creator(Role, name="Role", exclude=())
RoleBasic = pydantic_model_creator(Role, name="RoleIn", exclude_readonly=True)


class RoleIn(RoleBasic):
    menus:List[int]

class MenuList(BaseModel):
    rightId:List[int]


#class RoleQuery(BasicQuery):
    #name: str = None

class RoleQuery(BaseModel):
    name: str = None


MenuOut = pydantic_model_creator(Menu, name="Menu", exclude=())
MenuBasic = pydantic_model_creator(Menu, name="MenuIn", exclude_readonly=True)


class MenuIn(MenuBasic):
    meta: dict = None


DictOut = pydantic_model_creator(Dict, name="Dict")
DictBasic = pydantic_model_creator(Dict, name="DictIn", exclude_readonly=True)


class DictIn(DictBasic):
    value: dict = None


AuctionOut = pydantic_model_creator(Auction, name="Auction")
AuctionBasic = pydantic_model_creator(Auction, name="AuctionIn", exclude_readonly=True)


class AuctionIn(AuctionBasic):
    amount: float = None

class AuctionQuery(BaseModel):
    name: str = None

SymbolOut = pydantic_model_creator(Symbol, name="Symbol")
SymbolBasic = pydantic_model_creator(Symbol, name="SymbolIn", exclude_readonly=True)


class SymbolIn(SymbolBasic):
    name: str = None


KpiOut = pydantic_model_creator(Kpi, name="Kpi")
KpiBasic = pydantic_model_creator(Kpi, name="KpiIn", exclude_readonly=True)


class KpiIn(KpiBasic):
    kpi_name: str = None


dbtradedataOut = pydantic_model_creator(Dbtradedata, name="dbtradedata")
dbtradedataBasic = pydantic_model_creator(Dbtradedata, name="dbtradedataIn", exclude_readonly=True)
