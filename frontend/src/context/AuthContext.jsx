import React, { createContext, useState, useEffect } from 'react';
import axios from 'axios';

export const AuthContext = createContext();

const parseJwt = (token) => {
    try {
        return JSON.parse(atob(token.split('.')[1]));
    } catch (e) {
        return null;
    }
};

const AuthProvider = ({ children }) => {
    const [auth, setAuth] = useState({
        token: localStorage.getItem('token') || '',
        user: null,
    });

    useEffect(() => {
        if (auth.token) {
            // Decode token to get user info
            try {
                const payload = parseJwt(auth.token);
                if (payload) {
                    setAuth((prev) => ({
                        ...prev,
                        user: { username: payload.sub, tier: payload.tier },
                    }));
                } else {
                    setAuth({ token: '', user: null });
                }
            } catch (e) {
                console.error('Failed to parse token', e);
                setAuth({ token: '', user: null });
            }
        }
    }, [auth.token]);

    const login = async (username, password) => {
        try {
            const formData = new URLSearchParams();
            formData.append('username', username);
            formData.append('password', password);

            console.log('Attempting login...');

            const response = await axios.post('http://localhost:8000/auth/login', formData.toString(), {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            });

            console.log('Login response:', response.data);

            localStorage.setItem('token', response.data.access_token);
            setAuth({
                token: response.data.access_token,
                user: { username, tier: parseJwt(response.data.access_token).tier },
            });

            console.log('Auth state updated:', { token: response.data.access_token, username, tier: parseJwt(response.data.access_token).tier });
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    };

    const signup = async (username, email, password, tier) => {
        try {
            const response = await axios.post('http://localhost:8000/auth/signup', {
                username,
                email,
                password,
                tier
            }, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            // Optionally, you can auto-login the user after signup
            // Or redirect them to the login page
        } catch (error) {
            console.error('Signup error:', error);
            if (error.response) {
                throw new Error(error.response.data.detail || 'Signup failed');
            } else if (error.request) {
                throw new Error('No response received from the server');
            } else {
                throw new Error('Error setting up the request');
            }
        }
    };

    const logout = () => {
        localStorage.removeItem('token');
        setAuth({ token: '', user: null });
    };

    return (
        <AuthContext.Provider value={{ auth, login, logout, signup }}>
            {children}
        </AuthContext.Provider>
    );
};

export default AuthProvider;