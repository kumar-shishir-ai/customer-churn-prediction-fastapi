from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
from app.api import routes_auth, routes_predict
from app.middleware.logging_middleware import LoggingMiddleware
from app.core.exceptions import register_exception_handlers
from app.services.model_service import churn_prediction

app = FastAPI()

# Static folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/", response_class=HTMLResponse)
async def predict_home(request: Request):
    form_data = await request.form()
    data_dict = dict(form_data)
    
    try:
        data_dict["SeniorCitizen"] = int(data_dict.get("SeniorCitizen", 0))
        data_dict["MonthlyCharges"] = float(data_dict.get("MonthlyCharges", 0))
        data_dict["TotalCharges"] = float(data_dict.get("TotalCharges", 0))
        
        tenure = int(data_dict.get("tenure", 0))
        if tenure <= 12:
            data_dict["tenure_group"] = "1 - 12"
        elif tenure <= 24:
            data_dict["tenure_group"] = "13 - 24"
        elif tenure <= 36:
            data_dict["tenure_group"] = "25 - 36"
        elif tenure <= 48:
            data_dict["tenure_group"] = "37 - 48"
        elif tenure <= 60:
            data_dict["tenure_group"] = "49 - 60"
        else:
            data_dict["tenure_group"] = "61 - 72"
            
        if "tenure" in data_dict:
            del data_dict["tenure"]
            
        prediction_val = churn_prediction(data_dict)
        prediction_text = "Yes (Churn)" if prediction_val >= 0.5 else "No (Retain)"
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": prediction_text,
                "probability": float(prediction_val)
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": f"Error: {str(e)}",
                "probability": 0.0
            }
        )

# link middleware
app.add_middleware(LoggingMiddleware)

# Link endpoints
app.include_router(routes_auth.router, tags=['Auth'])
app.include_router(routes_predict.router, tags=['Prediction'])

# monitoring using Prometheus
Instrumentator().instrument(app).expose(app)

# Add exception handlers
register_exception_handlers(app)