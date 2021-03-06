import random

capitals= {"Alabama": "Montgomey", "Alaska": "Junea", "Arizona": "Phoenix",
 "Arkansas": "Little Rock", "California": "Sacramento", "Colorado": "Denver",
 "Conneticut": "Harford", "Delaware": "Dover", "Florida": "Tallahassee",
 "Georgia": "Atlanta","Hawai": "Honolulu", "Idaho": "Boise",
 "Illinois": "Springfield", "Indiana": "Indianapolis", "Iowa": "Des Moines",
 "Kansas": "Topeka", "Kentucky": "Frankfort","Louisiana": "Baton Rouge",
 "Maine": "Augusta","Maryland": "Annapolis", "Massachusetts": "Boston",
 "Michigan": "Lansing", "Minnesota": "Saint Paul", "Mississippi": "Jackson",
 "Missouri": "Jefferson City", "Montana": "Helena", "Nebraska": "Lincoin",
 "Nevada": "Carson City", "New Hampshire": "Concord", "New Jersey": "Trenton",
 "New Mexico": "Santa Fe", "New York": "Albany", "North Carolina": "Raleigh",
 "North Dakota": "Bismarck", "Ohio": "Columbus", "Oklahama": "Oklahama City",
 "Oregon": "Salem","Pennsylvania": "Harrisburg", "Rhode Island": "Providence",
 "South Carolina": "Columbia","South Dakota": "Pierre", "Tennessee": "Nashville",
 "Texas": "Austin", "Utah": "Salt Lake City","Vermont": "Montpelier","Virginia": "Richmond",
 "Washinton": "Olympia", "West Verginia": "Chareston","Wisconsin": "Madison","Wyoming":"Cheyenne"}
    #Generate 35 quiz files(35)
for quizNum in range(35):
    #TODO:Create the quiz and answer key files.
    #TODO:Write out the header for the quiz.
    #TODO:Shuffle the order of the states.
    #TODO:Loop through all 50 states, making a question for each.
    # Create the quiz and answer key files.
    quizFile = open("capitalsquiz{}.txt" .format(quizNum + 1), "w")
    answerKeyFile = open("capitalsquiz_answers{}.txt" .format(quizNum + 1), "w")
    # Write out the header for the quiz.
    quizFile.write("Name:\n\nDate:\n\nPeriod:\n\n")
    quizFile.write((" " * 20) + "State Capitals Quiz(From {})".format(quizNum +1))
    quizFile.write("\n\n")

   #Shuffle the order of the states.
    states = list(capitals.keys())
    random.shuffle(states)

  #TODO: Loop through all 50 states,making a question for a each.
    for questionNum in range(50):
    # Get right and wrong answers.
        correctAnswer = capitals[states[questionNum]]
        wrongAnswers = list(capitals.values())
        del wrongAnswers[wrongAnswers.index(correctAnswer)]
        wrongAnswers = random.sample(wrongAnswers, 3)
        answerOptions = wrongAnswers + [correctAnswer]
        random.shuffle(answerOptions)

   #TODO: Write the question and answer options to the quiz file.
        quizFile.write("{}. What is the capital of {} ?\n" .format(questionNum + 1, states[questionNum]))
        for i in range(4):
            quizFile.write(" {}. {}\n" .format("ABCD"[i], answerOptions[i]))
        quizFile.write("\n")
   #TODO: Write the answer key to a file.
        answerKeyFile.write("{}.{}\n" .format(questionNum + 1, "ABCD"[answerOptions.index(correctAnswer)]))
    quizFile.close()
    answerKeyFile.close()

