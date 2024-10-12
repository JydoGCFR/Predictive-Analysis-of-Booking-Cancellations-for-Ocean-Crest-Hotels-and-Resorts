# **Predictive Analysis of Booking Cancellations for Ocean Crest Hotels and Resorts**

**Introduction**

This report outlines a predictive analysis aimed at minimizing booking cancellations for Ocean Crest Hotels and Resorts. Given the significant revenue impact of cancellations, the project focuses on understanding customer booking behaviors and developing strategies to optimize marketing and operations to reduce these cancellations.

**Business Problem**

Booking cancellations have been identified as a critical challenge for Ocean Crest, adversely affecting operational efficiency and revenue. Between 2014 and 2017, the hotel experienced approximately 42,000 cancellations, resulting in a revenue loss of around $16 million. The objective of this project is to analyze booking data to identify factors contributing to cancellations and to provide actionable strategies to mitigate them.

**Data Exploration**

**Dataset Overview**
The dataset comprises 113,421 observations with various predictor variables relevant to booking cancellations, including:

Customer Type
Lead Time
Arrival Date
Number of Weekend and Weekday Stays
Average Daily Rate (ADR)
Reservation Status
Special Requests
Deposit Type

**Key Findings from Data Exploration**

**Lead Time:** Longer lead times are associated with lower cancellation rates, indicating that customers who book well in advance are less likely to cancel.

**Deposit Type:** Bookings requiring a non-refundable deposit exhibit significantly lower cancellation rates compared to those with no deposit.

**Market Segmentation:** Online Travel Agencies (OTA) and group bookings demonstrate the highest cancellation rates, highlighting specific segments that require targeted interventions.

**Customer Segmentation**

**Customer Categories**

**Short-term, High-Value Bookers:**

Characteristics: Short lead time, high spending, and often make special requests.
Strategy: Craft premium services with exclusive offers and incentivize late bookings with discounts on non-refundable deposits.

**Moderate Lead Time, Moderate Value Bookers:**

Characteristics: Mid-range spending with moderate lead times and few special requests.
Strategy: Create bundled offers that include value-added services, such as discounted breakfast and free parking.

**Long-term, Budget-Conscious Bookers:**

Characteristics: Long lead times, low spending, often opt for refundable deposits.
Strategy: Offer incentives for early bookings with non-refundable deposits and create partnerships with offline travel agents.

**Clustering Analysis**

Using hierarchical clustering, three distinct customer segments were identified based on their booking behaviors:

Cluster 1: Short lead time, high ADR, and frequent special requests.
Cluster 2: Moderate lead time and value spending, with occasional group bookings.
Cluster 3: Long lead time, low ADR, with a preference for offline bookings.

**Predictive Modeling**

**Model Selection**
XGBoost was chosen for its effectiveness in classification tasks and its ability to handle large datasets efficiently. The model was tuned using cross-validation to prevent overfitting and maximize performance.

**Model Performance**

Test AUC: 0.9264
Accuracy: 85.88%
Sensitivity: 74.74%
Specificity: 92.44%
F1 Score: 79.63%

**Feature of Importance**

The most influential factors in predicting booking cancellations were identified as:

Deposit Type: Non-refundable deposits significantly reduce cancellation likelihood.
Lead Time: Longer lead times correlate with lower cancellation rates.

**Recommendations**

**Enhance Booking Policies:** Implement non-refundable deposit options to discourage cancellations.

**Targeted Marketing Campaigns:** Develop campaigns focused on segments identified through clustering, emphasizing the benefits of early booking and exclusive offers.

**Partnerships with Travel Agents:** Collaborate with online and offline travel agents to increase market reach and improve booking stability.

**Monitor Market Trends:** Continuously assess booking behaviors and market conditions to adapt strategies as needed.

**Conclusion**

The analysis and predictive modeling conducted for Ocean Crest Hotels and Resorts have provided valuable insights into booking cancellations. By understanding customer characteristics and employing targeted strategies, the hotel can significantly reduce cancellations and enhance revenue. Implementing the recommended strategies is expected to mitigate booking cancellations by 30% and could lead to a revenue increase of approximately $5 million.
