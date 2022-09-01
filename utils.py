from datetime import datetime, timedelta
from typing import Optional

from passlib.context import CryptContext
from jose import jwt, JWTError

from fastapi import Request, HTTPException, Depends
from tortoise import Tortoise

from models import User

KEY = "lLNiBWPGiEmCLLR9kRGidgLY7Ac1rpSWwfGzTJpTmCU"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
algorithm = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码 vs hash密码
    :param plain_password: 明文密码
    :param hashed_password: hash密码
    :return:
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    加密明文
    :param password: 明文密码
    :return:
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    生成token
    :param data: 字典
    :param expires_delta: 有效时间
    :return:
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, KEY, algorithm=algorithm)
    return encoded_jwt


async def auth_token(request: Request):
    """token 依赖项"""

    credentials_exception = HTTPException(status_code=401, detail="认证失败")
    token = request.headers.get("Authorization")
    if token:
        # 解密token

        try:
            payload = jwt.decode(token, KEY, algorithms=[algorithm])
        except JWTError:
            raise credentials_exception
        username: str = payload.get("sub", None)
        user = await User.filter(username=username).first()
        if user :
            return user
    raise credentials_exception


async def api_premiss(request: Request, user=Depends(auth_token)):
    """接口权限验证"""
    conn = Tortoise.get_connection("default")
    # todo 登录时 一并获取后(当前用户 当前使用角色)存储在redis中, 切换角色时修改redis该数据信息
    result = await conn.execute_query_dict(
        """select m.api, m.method FROM
         user_role as ur LEFT JOIN role_menu as rm ON ur.role_id = rm.role_id LEFT JOIN menu as m ON rm.menu_id = m.id
          WHERE ur.user_id = (?) AND ur.role_id = (?) and ur.status = 0 AND m.api NOTNULL AND m.method NOTNULL
          """,
        [user.id, request.headers.get("currentRole")],
    )
    if not {"api": request.url.path, "method": request.method.lower()} in result:
        raise HTTPException(status_code=403, detail="无权访问该接口")


def list_to_tree(
    menus, parent_flag: str = "parent_id", children_key: str = "children"
) -> list:
    """
    list 结构转 树结构
    :param menus: [{id:1, parent_id: 3}]
    :param parent_flag: 节点关系字段
    :param children_key: 生成树结构的子节点字段
    :return: list 类型的 树嵌套数据
    """ ""
    menu_map = {menu["id"]: menu for menu in menus}
    # print(menu_map)
    arr = []
    for menu in menus:
        # 有父级
        mid = menu[parent_flag]
        
        if mid :
        
            result = menu_map[mid].get(children_key)
           
            if result:
                result.append(menu)
            else:
                menu_map[mid][children_key] = []
                menu_map[mid][children_key].append(menu)
        else:
            arr.append(menu)
    return arr
