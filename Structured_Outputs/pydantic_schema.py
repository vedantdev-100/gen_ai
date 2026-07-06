from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str 
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5)
    
new_student = {'name' : 'nitish','age':32, "email": "v@gmail.com", "cgpa": 5}

student = Student(**new_student)

# print(type(student))
# print(student)

student_dict = dict(student)
student_json = student.model_dump_json()

print(student_dict['age'])
print(student_json)