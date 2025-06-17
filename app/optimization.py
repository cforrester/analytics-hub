import streamlit as st

def run():
    st.header("Prescriptive Optimization (Linear Programming)")
    st.write("Define a linear optimization problem to solve:")
    opt_type = st.radio("Objective type", ["Maximize", "Minimize"], index=0)
    objective = st.text_input("Objective function (e.g., 5x + 3y)")
    constraints_text = st.text_area("Constraints (one per line, e.g., 2x + 3y <= 10)")
    if st.button("Solve Optimization Problem"):
        if not objective:
            st.error("Please enter an objective function.")
            return
        import pulp
        # Determine optimization direction
        sense = pulp.LpMaximize if opt_type == "Maximize" else pulp.LpMinimize
        lp_prob = pulp.LpProblem("UserProblem", sense)
        # Parse objective
        # Extract variables and coefficients
        import re
        vars_dict = {}
        # Ensure all variables in objective have LpVariable instances
        for var in re.findall(r'[A-Za-z]+', objective):
            if var not in vars_dict:
                vars_dict[var] = pulp.LpVariable(var, lowBound=0)  # assume non-negativity
        # Build objective expression
        objective_expr = 0
        for term in re.findall(r'([+-]?\s*\d*\.?\d*)\s*([A-Za-z]+)', objective.replace(' ', '')):
            coeff_str, var = term
            coeff = float(coeff_str) if coeff_str not in ["", "+", "-"] else 1.0
            if coeff_str.strip().startswith('-'):
                # if just '-' it means -1
                coeff = -1.0 if coeff_str.strip() == '-' else coeff
            objective_expr += coeff * vars_dict[var]
        lp_prob += objective_expr
        # Parse constraints
        if constraints_text:
            for line in constraints_text.splitlines():
                if not line.strip():
                    continue
                # Identify constraint type and parse
                # Regex to split left and right at <=,>=,=
                m = re.split(r'(<=|>=|=)', line.replace(' ', ''))
                if len(m) != 3:
                    st.error(f"Could not parse constraint: '{line}'")
                    return
                left, op, right = m
                # Ensure all vars in left have LpVariable
                for var in re.findall(r'[A-Za-z]+', left):
                    if var not in vars_dict:
                        vars_dict[var] = pulp.LpVariable(var, lowBound=0)
                # Build left expression
                left_expr = 0
                for term in re.findall(r'([+-]?\s*\d*\.?\d*)\s*([A-Za-z]+)', left):
                    coeff_str, var = term
                    coeff = float(coeff_str) if coeff_str not in ["", "+", "-"] else 1.0
                    if coeff_str.strip().startswith('-'):
                        coeff = -1.0 if coeff_str.strip() == '-' else coeff
                    left_expr += coeff * vars_dict[var]
                # Parse right value
                try:
                    right_val = float(right)
                except:
                    st.error(f"Right-hand side of constraint '{line}' is not a number.")
                    return
                # Add constraint to problem
                if op == '<=':
                    lp_prob += (left_expr <= right_val)
                elif op == '>=':
                    lp_prob += (left_expr >= right_val)
                else:  # '='
                    lp_prob += (left_expr == right_val)
        # Solve the LP problem
        try:
            lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception as e:
            st.error(f"Solver error: {e}")
            return
        st.subheader("Optimization Result")
        if pulp.LpStatus[lp_prob.status] != 'Optimal':
            st.write(f"Solver Status: {pulp.LpStatus[lp_prob.status]}")
        else:
            # Display variable values
            solution = {v.name: v.value() for v in lp_prob.variables()}
            st.write("Optimal decision variables:", solution)
            st.write("Optimal objective value:", pulp.value(lp_prob.objective))
