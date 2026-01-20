# Load packages

# Declare function
def ProbabilityOfHypothesisGivenData(prior_probability_of_hypothesis_being_true,
                                     prior_probability_of_data_given_hypothesis_being_true,
                                     prior_probability_of_data_given_hypothesis_being_false,
                                     return_results=False,
                                     # Payoff estimates
                                     payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true=None,
                                     payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false=None,
                                     payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true=None,
                                     payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false=None):
    """
    Calculate posterior probabilities and the expected value of information using Bayes' Theorem.

    This function applies Bayes' Theorem to update the probability of a hypothesis being true
    based on new evidence (data). It calculates the posterior probability of the hypothesis
    given that the data was observed, as well as the posterior probability if the data
    was NOT observed. If payoff values are provided, it also conducts a decision analysis
    to determine the Expected Value of Information (EVOI), helping determine if acquiring
    the data is worth the associated costs.

    Bayesian updating and EVOI are essential for:
      * Clinical Diagnostics: Updating the probability of a disease given a test result (sensitivity/specificity).
      * Intelligence Analysis: Assessing the likelihood of a threat based on new signal intelligence.
      * Quality Engineering: Determining if additional destructive testing is cost-effective.
      * Epidemiology: Estimating true infection prevalence from imperfect screening data.
      * Cybersecurity: Refining the probability of a system compromise based on IDS alerts.
      * Financial Risk: Evaluating the value of a market research report before purchase.
      * Legal Analysis: Updating the probability of guilt/innocence given a new piece of forensic evidence.
      * Environmental Science: Assessing the value of additional soil sampling for contamination mapping.

    Parameters
    ----------
    prior_probability_of_hypothesis_being_true : float
        The initial belief (0 to 1) that the hypothesis is true before seeing the data.
    prior_probability_of_data_given_hypothesis_being_true : float
        The "likelihood" or probability of observing the data if the hypothesis were true (e.g., test sensitivity).
    prior_probability_of_data_given_hypothesis_being_false : float
        The probability of observing the data if the hypothesis were false (e.g., false positive rate).
    return_results : bool, optional
        Whether to return the calculated values as a dictionary. If False, the function
        only prints the results. Defaults to False.
    payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true : float, optional
        The value or utility gained if you correctly bet the hypothesis is true.
    payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false : float, optional
        The value or utility (often 0 or a penalty) if you bet true but the hypothesis is false.
    payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true : float, optional
        The value or utility if you bet false but the hypothesis is actually true.
    payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false : float, optional
        The value or utility gained if you correctly bet the hypothesis is false.

    Returns
    -------
    dict or None
        If `return_results` is True, returns a dictionary containing posterior probabilities
        and, if payoffs were provided, expected values and the Expected Value of Information.
        Otherwise, returns None and prints results to the console.

    Examples
    --------
    # Healthcare: Revising the probability of a rare disease after a positive test result
    # Disease prevalence (prior): 1%, Test Sensitivity: 95%, False Positive Rate: 5%
    results = ProbabilityOfHypothesisGivenData(
        prior_probability_of_hypothesis_being_true=0.01,
        prior_probability_of_data_given_hypothesis_being_true=0.95,
        prior_probability_of_data_given_hypothesis_being_false=0.05,
        return_results=True
    )

    # Intelligence: Evaluating if a $5,000 sensor report is worth the cost
    # Prior belief of threat: 20%, Sensor reliability (P(Data|True)): 80%, FPR: 10%
    # Payoffs: $100k for stopping a threat, $0 for false alarms/misses
    evoi_analysis = ProbabilityOfHypothesisGivenData(
        prior_probability_of_hypothesis_being_true=0.20,
        prior_probability_of_data_given_hypothesis_being_true=0.80,
        prior_probability_of_data_given_hypothesis_being_false=0.10,
        payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true=100000,
        payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false=0,
        payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true=0,
        payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false=100000,
        return_results=True
    )
    """

    # Ensure the prior probabilities are between 0 and 1
    if prior_probability_of_hypothesis_being_true < 0 or prior_probability_of_hypothesis_being_true > 1:
        raise ValueError("The prior probability of the hypothesis being true must be between 0 and 1.")
    if prior_probability_of_data_given_hypothesis_being_true < 0 or prior_probability_of_data_given_hypothesis_being_true > 1:
        raise ValueError("The prior probability of the data given the hypothesis is true must be between 0 and 1.")
    if prior_probability_of_data_given_hypothesis_being_false < 0 or prior_probability_of_data_given_hypothesis_being_false > 1:
        raise ValueError("The prior probability of the data given the hypothesis is false must be between 0 and 1.")

    # Ensure that if any payoffs are provided, all four are provided
    if payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true is not None and payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false is not None and payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true is not None and payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false is not None:
        if payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true < 0 or payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false < 0 or payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true < 0 or payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false < 0:
            raise ValueError("The payoffs must be non-negative.")

    # Calculate the prior probability of hypothesis not being true
    prior_probability_of_hypothesis_being_false = 1 - prior_probability_of_hypothesis_being_true
    # print("The prior probability of the hypothesis being false is:", prior_probability_of_hypothesis_being_false)

    # Calculate the posterior probability of observing the data
    p_of_data = (prior_probability_of_hypothesis_being_true * prior_probability_of_data_given_hypothesis_being_true) + (prior_probability_of_hypothesis_being_false * prior_probability_of_data_given_hypothesis_being_false)
    print("The posterior probability of the data (under the assumption that the data is a valid observation and not due to noise or error) is:", format(p_of_data, ".1%"))

    # Calculate the posterior probability of not observing the data
    p_of_not_data = 1 - p_of_data
    print("The posterior probability of the data NOT being observed (i.e., the data/evidence is not what you expected it to be) is:", format(p_of_not_data, ".1%"))

    # Calculate the posterior probability of the hypothesis given the data
    p_of_hypothesis_being_true_given_data = (prior_probability_of_hypothesis_being_true * prior_probability_of_data_given_hypothesis_being_true) / p_of_data
    print("\nThe posterior probability of the hypothesis being true given the data (under the assumption that the data is a valid observation and not due to noise or error) is:", format(p_of_hypothesis_being_true_given_data, ".1%"))

    # Calculate the posterior probability of the hypothesis given the data
    p_of_hypothesis_being_false_given_data = (prior_probability_of_hypothesis_being_false * prior_probability_of_data_given_hypothesis_being_false) / p_of_data
    print("The posterior probability of the hypothesis being false given the data (under the assumption that the data is a valid observation and not due to noise or error) is:", format(p_of_hypothesis_being_false_given_data, ".1%"))

    # Calculate the posterior probability of the hypothesis given the data
    p_of_hypothesis_being_true_given_not_data = (prior_probability_of_hypothesis_being_true * (1 - prior_probability_of_data_given_hypothesis_being_true)) / p_of_not_data
    print("\nThe posterior probability of the hypothesis being true given data NOT being observed (i.e., the data/evidence is not what you expected it to be) is:", format(p_of_hypothesis_being_true_given_not_data, ".1%"))

    # Calculate the posterior probability of the hypothesis given the data
    p_of_hypothesis_being_false_given_not_data = (prior_probability_of_hypothesis_being_false * (1 - prior_probability_of_data_given_hypothesis_being_false)) / p_of_not_data
    print("The posterior probability of the hypothesis being false given data NOT being observed (i.e., the data/evidence is not what you expected it to be) is:", format(p_of_hypothesis_being_false_given_not_data, ".1%"))

    # If payoff information provided....
    if payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true is not None and payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false is not None and payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true is not None and payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false is not None:
        # Calculate the value of bets on the hypothesis being true
        value_of_bet_on_hypothesis_being_true = (payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true * prior_probability_of_hypothesis_being_true) + (payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false * prior_probability_of_hypothesis_being_false)
        print("\nThe value of bets on the hypothesis being true is:", format(value_of_bet_on_hypothesis_being_true, ".2f"))

        # Calculate the value of bets on the hypothesis being false
        value_of_bet_on_hypothesis_being_false = (payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true * prior_probability_of_hypothesis_being_true) + (payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false * prior_probability_of_hypothesis_being_false)
        print("The value of bets on the hypothesis being false is:", format(value_of_bet_on_hypothesis_being_false, ".2f"))

        # Get the maximum value of bets
        max_value_of_bets = max(value_of_bet_on_hypothesis_being_true, value_of_bet_on_hypothesis_being_false)
        
        # Calculate the expected value of on the hypothesis being true given the data
        expected_value_of_bets_on_hypothesis_being_true_given_data = (payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true * p_of_hypothesis_being_true_given_data) + (payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false * p_of_hypothesis_being_false_given_data)
        print("\nThe expected value of bets on the hypothesis being true given the data (under the assumption that the data is a valid observation and not due to noise or error) is:", format(expected_value_of_bets_on_hypothesis_being_true_given_data, ".2f"))

        # Calculate the expected value of on the hypothesis being false given the data
        expected_value_of_bets_on_hypothesis_being_false_given_data = (payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true * p_of_hypothesis_being_true_given_data) + (payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false * p_of_hypothesis_being_false_given_data)
        print("The expected value of bets on the hypothesis being false given the data (under the assumption that the data is a valid observation and not due to noise or error) is:", format(expected_value_of_bets_on_hypothesis_being_false_given_data, ".2f"))

        # Get the maximum expected value of bets given the data
        max_expected_value_of_bets_given_data = max(expected_value_of_bets_on_hypothesis_being_true_given_data, expected_value_of_bets_on_hypothesis_being_false_given_data)

        # Calculate the expected value of on the hypothesis being true given data NOT being observed
        expected_value_of_bets_on_hypothesis_being_true_given_not_data = (payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true * p_of_hypothesis_being_true_given_not_data) + (payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false * p_of_hypothesis_being_false_given_not_data)
        print("\nThe expected value of bets on the hypothesis being true given data NOT being observed (i.e., the data/evidence is not what you expected it to be) is:", format(expected_value_of_bets_on_hypothesis_being_true_given_not_data, ".2f"))

        # Calculate the expected value of on the hypothesis being false given data NOT being observed
        expected_value_of_bets_on_hypothesis_being_false_given_not_data = (payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true * p_of_hypothesis_being_true_given_not_data) + (payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false * p_of_hypothesis_being_false_given_not_data)
        print("The expected value of bets on the hypothesis being false given data NOT being observed (i.e., the data/evidence is not what you expected it to be) is:", format(expected_value_of_bets_on_hypothesis_being_false_given_not_data, ".2f"))

        # Get the maximum expected value of bets given data NOT being observed
        max_expected_value_of_bets_given_not_data = max(expected_value_of_bets_on_hypothesis_being_true_given_not_data, expected_value_of_bets_on_hypothesis_being_false_given_not_data)

        # Calculate the expected value with data/information
        exp_value_without_data = max_value_of_bets
        exp_value_with_data = (max_expected_value_of_bets_given_data * p_of_data) + (max_expected_value_of_bets_given_not_data * (1 - p_of_data))

        # Calculate the expected value of information
        exp_value_of_information = exp_value_with_data - exp_value_without_data
        print("\nThe expected value of additional information is:", format(exp_value_of_information, ".2f"))


    # Return posterior probabilities as a dictionary
    if return_results:
        # Create dictionarity of posterior probabilities
        return_dict = {
            "Posterior probability of the hypothesis being true given the data": round(p_of_hypothesis_being_true_given_data, 4),
            "Posterior probability of the hypothesis being false given the data": round(p_of_hypothesis_being_false_given_data, 4),
            "Posterior probability of the hypothesis being true given data NOT being observed": round(p_of_hypothesis_being_true_given_not_data, 4),
            "Posterior probability of the hypothesis being false given data NOT being observed": round(p_of_hypothesis_being_false_given_not_data, 4),
        }

        # Add expected value of information if payout information is provided
        if payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_true is not None and payoff_if_bet_on_hypothesis_being_true_and_hypothesis_is_false is not None and payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_true is not None and payoff_if_bet_on_hypothesis_being_false_and_hypothesis_is_false is not None:
            return_dict["Expected value of payoffs on the hypothesis being false given the data"] = round(expected_value_of_bets_on_hypothesis_being_false_given_data, 2)
            return_dict["Expected value of payoffs on the hypothesis being true given the data"] = round(expected_value_of_bets_on_hypothesis_being_true_given_data, 2)
            return_dict["Expected value of payoffs on the hypothesis being false given data NOT being observed"] = round(expected_value_of_bets_on_hypothesis_being_false_given_not_data, 2)
            return_dict["Expected value of payoffs on the hypothesis being true given data NOT being observed"] = round(expected_value_of_bets_on_hypothesis_being_true_given_not_data, 2)
            return_dict["Expected value of payoffs without data"] = round(exp_value_without_data, 2)
            return_dict["Expected value of payoffs with data"] = round(exp_value_with_data, 2)
            return_dict["Expected value of information"] = round(exp_value_of_information, 2)
        
        # Return dictionary
        return return_dict

