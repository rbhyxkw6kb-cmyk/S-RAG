from secrets import compare_digest

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.config import settings

basic = HTTPBasic(realm="Zhiyuan", auto_error=False)


def access_role(credentials: HTTPBasicCredentials | None = Depends(basic)) -> str:
    if settings.staff_password is None and settings.admin_password is None:
        return "admin"
    if credentials:
        roles = {"staff": settings.staff_password, "admin": settings.admin_password}
        password = roles.get(credentials.username)
        if password and compare_digest(credentials.password.encode(), password.get_secret_value().encode()):
            return credentials.username
    raise HTTPException(401, "请先登录。", headers={"WWW-Authenticate": 'Basic realm="Zhiyuan"'})


def require_admin(role: str = Depends(access_role)) -> None:
    if role != "admin":
        raise HTTPException(403, "仅知识库管理员可以导入指导书。")
