from fastapi import APIRouter

itemRouter=APIRouter()

@itemRouter.get("/")
def create_user():
     return 'item'