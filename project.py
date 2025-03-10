import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split ,cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.express as px
import streamlit as st
import joblib
Scaler=joblib.load('sc.pkl')
model=joblib.load('model.pkl')
import streamlit as st  
import time

# Set the title  
st.title("🤖 Smart Order Receiving & Inventory Optimization with AI")  

# Set the subtitle (description)  
st.markdown(  
    "📊 **Enhancing Demand Forecasting, Reducing Waste, and Maximizing Operational Efficiency with Machine Learning.**"  
)  

st.divider()

current_balance = st.text_input('Please Enter Your Current Balance of the Granola Item')

# Convert input to integer only if it's not empty
if current_balance :
    try:
        current_balance = int(current_balance)
        x=[current_balance]
        if current_balance in np.arange(151): 
            st.divider()
            x1=np.array(x)
            x_array=Scaler.transform([x1])
            prediction=model.predict(x_array)[0]
            progress_bar = st.progress(0)
            if st.button("📊 Get Prediction"):
                with st.spinner("🔄 Processing..."):
                    time.sleep(2)
                    for i in range(100):
                        time.sleep(0.02)  # Simulate progress step
                        progress_bar.progress(i + 1)
                st.success(f"🔮 The predicted order quantity is: **{round(prediction,2)}**")
            st.divider()
            import plotly.graph_objects as go
            import numpy as np
            df=pd.read_excel(r'the branch.xlsx')
            df_numeric=df.select_dtypes(int)
            x=df_numeric.loc[:,'current_balance'].values
            y=df_numeric.iloc[:,-1].values
            x=x.reshape(-1,1)
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
            sc=StandardScaler()
            x_train_Scaled=sc.fit_transform(x_train)
            x_test_Scaled=sc.transform(x_test)
            y_pred=model.predict(x_test_Scaled)

            # Combine all x-values (scaled)
            x_all = np.concatenate([x_train_Scaled, x_test_Scaled])

            # Convert scaled values back to original (unscaled)
            x_all_nes = np.concatenate([
                sc.inverse_transform(x_train_Scaled), 
                sc.inverse_transform(x_test_Scaled)
            ])

            # Make predictions
            y_all_pred = model.predict(x_all)

            # Sort values for a smooth regression line
            sorted_indices = np.argsort(x_all_nes.flatten())  # Sort indices based on unscaled values
            x_sorted = x_all_nes.flatten()[sorted_indices]    # Sorted unscaled x-values
            y_sorted = y_all_pred[sorted_indices]             # Sorted predictions

            # Create a Plotly figure
            fig = go.Figure()

            # 🔹 Scatter plot for actual order receiving (Unscaled)
            fig.add_trace(go.Scatter(
                x=sc.inverse_transform(x_train_Scaled).flatten(), 
                y=y_train, 
                mode='markers', 
                marker=dict(color='blue', size=8, symbol='circle', opacity=0.8),  # Larger markers
                name="Actual Order Receiving",
                hovertemplate="Balance: %{x}<br>Actual Orders: %{y}<extra></extra>"
            ))
            # 🔹 Regression Line (Unscaled)
            fig.add_trace(go.Scatter(
                x=x_sorted, 
                y=y_sorted, 
                mode='lines', 
                line=dict(color='red', width=3, dash='solid'),  # Smoother line
                name="Predicted Order Receiving",
                hovertemplate="Balance: %{x}<br>Predicted Orders: %{y}<extra></extra>"
            ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text="Comparison: Actual vs. Predicted Order Receiving",
                    x=0.5, xanchor='center', font=dict(size=16)
                ),
                xaxis=dict(
                    title="📦 Current Balance",
                    showgrid=True, gridcolor='lightgray',
                    zeroline=True, zerolinecolor='black'
                ),
                yaxis=dict(
                    title="📈 Order Receiving",
                    showgrid=True, gridcolor='lightgray'
                ),
                legend=dict(
                    x=0.01, y=1.01, bordercolor="Black", borderwidth=1
                ),
                template="plotly_white",
                hovermode="x unified"  # Tooltip follows x-axis
            )
            st.plotly_chart(fig)

            st.markdown("""
            ### 📈 Key Takeaways:
            - 🔍 This model accurately predicts order quantities based on current inventory.
            - 📊 Helps **prevent overstocking and reduce waste**, ensuring **optimized inventory levels**.
            - 🚀 Businesses can use these insights to **automate ordering and streamline operations**.
            """)
            st.divider()
            st.title('The Problem: Inaccurate Item Orders 🛑 and Its Devastating Impact on Inventory and Operations 😞')
            st.markdown(
    "<h3 style='text-align: center; color: #FF4B4B;'>📉 Error Analysis: Comparing Sum of Squared Errors in AI-Powered vs. Mean-Based Ordering</h3>", 
    unsafe_allow_html=True

            )

            y_test_mean = np.mean(y_test)

            # Compute Errors (Difference from Mean)
            errors = y_test - y_test_mean

            # Compute Sum of Squared Errors (SSE)
            sse = np.sum(errors**2)

            # Create the scatter plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x_test, y_test, label="Actual Data (y_test)", color='blue', marker='o')

            # Plot the Mean Line
            ax.axhline(y_test_mean, color='red', linestyle='dashed', label="Mean of y_test")

            # Add error bars (Lines from each actual point to the mean)
            for i in range(len(x_test)):
                ax.plot([x_test[i], x_test[i]], [y_test[i], y_test_mean], 'gray', linestyle='dotted')  # Error lines
                ax.text(x_test[i], (y_test[i] + y_test_mean) / 2, f"{errors[i]:.2f}", 
                        fontsize=10, color="black", ha='center', bbox=dict(facecolor='white', alpha=0.5))  # Error values

            # Add SSE annotation
            ax.text(max(x_test), max(y_test), f"SSE: {sse:.2f}", 
                    fontsize=12, color="black", bbox=dict(facecolor='yellow', alpha=0.5))

            # Labels and title
            ax.set_xlabel("Current Balance")
            ax.set_ylabel("Order Receiving")
            ax.set_title("📊 Data Distribution with Mean Line, Errors, and SSE",fontname="Segoe UI Emoji") 
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.write(f"**Sum of Squared Error (SSE) for the Mean Model:** {sse:.2f} 💡")
            st.write("""
            The **Sum of Squared Error (SSE)** measures the total deviation between the predicted and actual values. 
            A **lower SSE** means better accuracy and model fit.🎯
            """)
            st.write("### Model Evaluation Metrics:")
            st.write("#### R² Score (Accuracy Ratio):")
            y_test_pred_mean = np.full_like(y_test, y_test_mean)

            # Display R² score for mean model predictions
            r2 = r2_score(y_test, y_test_pred_mean)
            st.write(f"**R² Score:** {r2:.2f} 🚀")
            st.write("""
            The **R² score** shows how well the model's predictions match the actual data. 
            Aim high! 💪
            """)

            st.divider()
            st.title('Predicting the Ordering Patterns of Items in a Restaurant: A Game-Changing Solution to Revolutionize Inventory Management 🚀')

            st.markdown(
    "<h2 style='text-align: center; color: #FF4B4B;'>🔥 “AI-Powered Forecasting Reduces Errors and Optimizes Inventory!” 🔥</h2>", 
    unsafe_allow_html=True
)

            errors = y_test - y_pred

        # Compute Sum of Squared Errors (SSE)
            sse = np.sum(errors**2)

            # Create the scatter plot
            fig,ax=plt.subplots(figsize=(8, 5))
            ax.scatter(x_test, y_test, label="Actual Data (y_test)", color='blue', marker='o')
            ax.scatter(x_test, y_pred, label="Predicted Data (y_pred)", color='red', marker='x')

            # Plot the prediction line
            ax.plot(x_test, y_pred, color='red', linestyle='dashed', label="Prediction Line (y_pred)")

            # Add error bars (Lines from each actual point to predicted point)
            for i in range(len(x_test)):
                ax.plot([x_test[i], x_test[i]], [y_test[i], y_pred[i]], 'gray', linestyle='dotted')  # Error lines
                ax.text(x_test[i], (y_test[i] + y_pred[i])/2 , f"{errors[i]:.2f}", 
                        fontsize=10, color="black", ha='center', bbox=dict(facecolor='white', alpha=0.5))  # Error values

            # Add SSE annotation
            # ax.text(max(x_test), max(y_test), f"SSE: {sse:.2f}", 
            #         fontsize=12, color="black", bbox=dict(facecolor='yellow', alpha=0.5))
            ax.text(max(x_test) - 10, max(y_test) - 10, f"SSE: {sse:.2f}", 
                    fontsize=12, color="black", bbox=dict(facecolor='yellow', alpha=0.5))

            # Labels and title
            ax.set_xlabel("Current Balance")
            ax.set_ylabel("Order recieving")
            ax.set_title("📊 Data Distribution with prediction Line, Errors, and SSE", fontname="Segoe UI Emoji")  
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            # Assuming sse, y_test, and y_pred are already defined
            st.write(f"**Sum of Squared Error (SSE) for the Mean Model:** {sse:.2f} 💡")
            st.write("""
            The **Sum of Squared Error (SSE)** quantifies the error between the predicted and actual values. 
            A **lower SSE** indicates better model performance as the predictions are closer to the actual values .🎯
            """)
            st.write("### Model Accuracy:")
            st.write(f"**R² Score (Accuracy of the Prediction):** {r2_score(y_test, y_pred):.2f} 🚀")
            st.write("""
            The **R² score** shows how well the model's predictions match the actual data. 
            . Great job 💪
            """)

            st.markdown("""
            ### 📌Key Takeaways:
            - ✅ AI-based predictions help **reduce inventory waste** by making **accurate demand forecasts**.  
            - 📉 **Lower SSE (Sum of Squared Errors) means AI predictions are more precise** than the traditional mean-based approach.  
            - ⚙️ Automating stock predictions **optimizes ordering efficiency**, reducing overstock and shortages.
            """)
            st.divider()
            st.markdown("## 🚀 Revolutionizing Inventory Management with AI! 🤖")  
            st.write(
                "Inaccurate order predictions can lead to **wasted inventory, financial losses, and operational inefficiencies**. ❌ "
                "Restaurants often struggle with stock shortages during peak hours or excessive waste from overstocking perishable items."
            )

            st.write(
                "To tackle this challenge, this **Smart Order Receiving & Inventory Optimization AI** was developed to "
                "**predict item ordering patterns with precision and efficiency**. 📊✅"
            )

            st.markdown("### 🔥 Key Benefits:")
            st.markdown("- 🔹 **Reduces waste** by optimizing inventory levels.")
            st.markdown("- 🔹 **Prevents stock shortages**, ensuring customer satisfaction.")
            st.markdown("- 🔹 **Enhances decision-making** with AI-driven insights.")

            st.write('''This is just the beginning! I'm excited about the potential of AI in
                      **transforming operational efficiency**. If you're interested in **AI-driven
                      business solutions**, let's connect and discuss! 💡''')
    except:
          st.error("⚠️ Please enter a valid number.")
          current_balance = None  # Reset value if invalid
