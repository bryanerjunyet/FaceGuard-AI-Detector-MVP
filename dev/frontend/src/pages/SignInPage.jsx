import { useState } from "react";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { signIn } from "@/api/client";

export default function SignInPage({ onSignInSuccess }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();

    const formData = new FormData(event.currentTarget);
    const submittedEmail = String(formData.get("email") || "").trim();
    const submittedPassword = String(formData.get("password") || "");

    if (!submittedEmail || !submittedPassword) {
      setIsError(true);
      setMessage("Email and password are required.");
      return;
    }

    setMessage("");
    setIsError(false);
    setIsSubmitting(true);

    try {
      const response = await signIn({ email: submittedEmail, password: submittedPassword });
      setMessage(response.message || "Sign in successful.");
      onSignInSuccess?.();
      setEmail(submittedEmail);
      setPassword("");
      navigate("/", { replace: true });
    } catch (error) {
      setIsError(true);
      setMessage(error.message || "Unable to sign in.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="page">
      <section className="panel panel-form">
        <h2>Sign In (Prototype)</h2>
        <form onSubmit={handleSubmit}>
          <label htmlFor="email">Email</label>
          <input
            id="email"
            name="email"
            type="email"
            placeholder="name@example.com"
            required
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
          <label htmlFor="password">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            placeholder="********"
            required
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
          <button className="primary-btn" type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Signing In..." : "Sign In"}
          </button>
        </form>
        {message ? <p className="panel-note">{isError ? `Error: ${message}` : message}</p> : null}
        <p className="auth-switch">
          Need an account? <Link to="/signup">Sign Up</Link>
        </p>
      </section>
    </main>
  );
}

