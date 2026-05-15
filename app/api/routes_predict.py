from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.dependencies import get_current_user, get_api_key
from app.services.model_service import churn_prediction


router = APIRouter()

class ChurnFeatures(BaseModel):
    gender: str 
    SeniorCitizen: int
    Partner: str 
    Dependents: str 
    PhoneService: str 
    MultipleLines: str 
    InternetService: str 
    OnlineSecurity: str 
    OnlineBackup: str 
    DeviceProtection: str 
    TechSupport: str 
    StreamingTV: str 
    StreamingMovies: str 
    Contract: str 
    PaperlessBilling: str
    PaymentMethod: str 
    MonthlyCharges: float
    TotalCharges: float
    tenure_group: str 

@router.post("/predict")
def predict_churn(churn: ChurnFeatures, user = Depends(get_current_user), _=Depends(get_api_key)):
    prediction = churn_prediction(churn.model_dump())
    return {"predicted_churn": f'{prediction:,.2f}'}
