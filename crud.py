from typing import Type, Generic

from tortoise.contrib.pydantic import PydanticModel
from tortoise.models import MODEL


class BaseCrud:

    @classmethod
    def page_query(cls, model: Type[MODEL]):
        """
       
        :param model: 模型类
        :return:
        """
        
        return model.filter(state__not=9).all()

    @classmethod
    async def delete_obj(cls, model: Type[MODEL], pk: int):
        return model.filter(id=pk).update(state=9)

    @classmethod
    async def m2m_create(cls, model: Type[MODEL], form: PydanticModel, *, m2: Type[MODEL],
                         del_field: str, m2_field: str):
        """
        多对多新增操作
        :param model:
        :param form: 用户pydantic入参
        :param m2: 关系模型
        :param del_field: 用户入参中的数据字段 一般是 [mid,mid1,...]
        :param m2_field: orm 定义的关联字段 一般是复数
        :return: 返回新增的数据对象
        example:

        class Role(BaseModel, Mixin):
            name = fields.CharField(max_length=20, description='角色名称')
            remark = fields.CharField(max_length=100, description='角色描述', null=True)
            menus: fields.ManyToManyRelation["Menu"] = fields.ManyToManyField(
                "models.Menu", related_name="roles", through="role_menu"
            )

        class Menu(BaseModel, Mixin):
            name = fields.CharField(max_length=20, description='名称', null=True)
            meta = fields.JSONField(description='元数据信息', null=True)
            path = fields.CharField(max_length=128, description='菜单url', null=True)
            type = fields.IntEnumField(MenuType, description='菜单类型', null=True)
            component = fields.CharField(max_length=128, description='组件地址从views目录开始', null=True)
            parent = fields.ForeignKeyField('models.Menu',
            on_delete=fields.SET_NULL, related_name='children', null=True)
            roles: fields.ManyToManyRelation[Role]

        新增角色：
            dao = BaseCrud(Role)
            role_obj = await dao.m2m_create(form,
            m2=Menu, del_field='menus', m2_field='menus')

        """
        m2_objs = [await m2.get(id=item)
                   for item in getattr(form, del_field)]
        delattr(form, del_field)
        m_obj = await model.create(**form.dict())
        await getattr(m_obj, m2_field).add(*m2_objs)
        return m_obj

    @classmethod
    async def m2m_update(cls, pk: int, model: Type[MODEL], form: PydanticModel, m2: Type[MODEL],
                         del_field: str, m2_field: str):
        """
        多对多修改信息
        :param pk: 修改对象的主键
        :param model: 模型类
        :param form: 输入模型
        :param m2: 关系模型
        :param del_field:输入模型中关系字段一般是个列表，需要删除
        :param m2_field: 关系模型字段，一般和上面的字段一样
        :return: 修改前的对象数据信息
        """
        m2_objs = [await m2.get(id=item) for item in getattr(form, del_field)]
        delattr(form, del_field)
        m_obj = await model.get(pk=pk)
        await getattr(m_obj, m2_field).clear()
        await getattr(m_obj, m2_field).add(*m2_objs)
        return m_obj
