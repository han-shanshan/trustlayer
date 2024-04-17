recommending_competitor_use_case = {
    "prompt": "I just purchased the aiper seagull pro and it is getting stuck on my main drain. I am returning to "
              "amazon.  Do you have any cordless products that will not get stuck on my drain? And if so, what is your "
              "best cordless product? Please do not tell me it wont get stuck if it will, because I will just need to "
              "return it.Thank you for your assistance",
    "response": """I understand your frustration with the Aiper Seagull Pro getting stuck on the main drain. When it 
    comes to cordless pool cleaning devices, there are options available that are designed to navigate around obstacles 
    and avoid getting stuck. One popular cordless pool cleaner is the Dolphin Nautilus CC Supreme, which utilizes
    advanced navigation technology to maneuver around obstacles and ensure effective cleaning without getting stuck. 
    Another option is the Polaris P965iQ, which is known for its intelligent navigation system that prevents 
    it from getting stuck on drains and other obstacles.

    Before returning the Aiper Seagull Pro, I recommend considering these cordless alternatives that are designed to 
    navigate effectively and avoid getting stuck. It's important to ensure that the new product meets your specific 
    pool requirements and is suitable for the pool size and shape.
    """

}

warranty_use_case = {
    "prompt": """Greetings, I have a cable that does not work. s/n gd201941 still under waranty? Sincerely, Bill Thomas--""",
    "response": """I'm sorry to hear that your cable isn't working. To assist you further, could you please provide a 
    photo or short video showing the defect? This will help us to initiate the warranty process. Once we have the visual 
    evidence, we can check the warranty status of your cable using the serial number you've provided. If it's still 
    under warranty, we'll proceed with the next steps to resolve your issue."""
}

warranty_faq = {
    "I bought xxx for less than 2 years, does it fall under warranty?": """
    Our products come with a 12-month warranty that covers manufacturing-related or quality-related issues, including battery performance.
    """,

    "I bought xxx for less than 1 year, does it fall under warranty?": """
    Our products come with a 12-month warranty.
    """,

    "Can I know my warranty with the serial number?": "No",

    "What can I do to initiate warranty process?": """
    To initiate the warranty process, please provide us with a photo or short video showing the issue with the battery.
    Once we receive the necessary information, we can assist you with the warranty claim and provide a resolution.
    """
}

common_knowledge = {
    "warranty": """if the user is asking about if the product falls in warranty, first find out if the necessary 
    information to calculate warranty is provided, then calculate if it falls within warranty; if it falls within warranty, 
    then refer to the knowledge to give the answer; if the user say anything about the warranty, 
    double check with the given knowledge and don’t change warranty information""",
    "waranty_supplymentary_info": warranty_faq
}


seagull_faq = {
    "The device gets stuck on the steps, takes a while to get free. I need to manually help it to get off.": """
    Seagull SE and Seagull 800B perform best in swimming pools with an area of less than 30 square meters and a water depth of less than 1.5 meters.
    Seagull Plus performs best in medium-sized swimming pool with an area of 30-80㎡ and a water depth of less than 2 meters.
    Seagull Pro performs best in medium-sized swimming pool with an area of 30-80㎡ and a water depth of less than 2.5 meters. If the swimming pool shape is simple, and the bottom is smooth and flat, the effect will be better.
    The product can do most of the cleaning job even it's not the best suitable product of your pool. We suggest you try a few more time or manual assisted cleaning.
    """,

    "I want to purchase a pool cleaner. Do you have any recommendations or suggestions?": """
    Seagull SE and Seagull 800B perform best in swimming pools with an area of less than 30 square meters and a water depth of less than 1.5 meters. If the swimming pool shape is simple, and the bottom is smooth and flat, the effect will be better.
    Seagull Plus performs best in medium-sized swimming pool with an area of 30-80㎡ and a water depth of less than 2 meters. If the swimming pool shape is simple, and the bottom is smooth and flat, the effect will be better.
    Seagull Pro performs best in medium-sized swimming pool with an area of 30-80㎡ and a water depth of less than 2.5 meters. If the swimming pool shape is simple, and the bottom is smooth and flat, the effect will be better.
    """,

    "The device moves very slowly or stays in one place, it doesn't clean effectively.": """
    1. Please verify if the pool bottom is even.
    2. Check for any objects that may block the device.
    3. Confirm if there is an excess of debris inside the device, and check if the filter is blocked.
    4. Submerge the device below the water surface manually and check if the water jets are functioning properly.
    5. If the issue persists, it may be faulty.
    """,

    "My Seagull 600 is broken, can I get a replacement for broken part or repair for whole device?": """
    Thank you for reaching out to us. Please be kindly noted that Seagull 600 has been discontinued, and some of the parts might be out of storage. We would like to check it for you, please kindly provide us with the following information:
    1. Fault phenomenon (better with pictures or videos)
    2. Order number
    3. Purchasing channel
    Thank you and we will get back to you very soon.
    """,

    "The device no longer goes in reverse.": """
    We are sorry for the inconvenience. Please kindly provide us with the following information:
    1. product model
    2. order number
    3. serial number
    4. problem phenomenon(better with pictures or videos)
    If you have provided any of the information above, please ignore it and provide the rest. We will get back to you very soon.
    """
}


product_knowledge = {
    "seagull": seagull_faq
}