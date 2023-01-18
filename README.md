# -employee-management-mobile-application
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const Employee = require('./models/employee');

const app = express();
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost/employee-management', { useNewUrlParser: true });

// Add Employee
app.post('/employees', (req, res) => {
    const employee = new Employee(req.body);
    employee.save((err, employee) => {
        if (err) res.status(500).send(err);
        res.status(201).json(employee);
    });
});

// List Employees
app.get('/employees', (req, res) => {
    Employee.find({}, (err, employees) => {
        if (err) res.status(500).send(err);
        res.json(employees);
    });
});

// Update Employee
app.put('/employees/:id', (req, res) => {
    Employee.findByIdAndUpdate(req.params.id, req.body, { new: true }, (err, employee) => {
        if (err) res.status(500).send(err);
        res.json(employee);
    });
});

// Delete Employee
app.delete('/employees/:id', (req, res) => {
    Employee.findByIdAndRemove(req.params.id, (err) => {
        if (err) res.status(500).send(err);
        res.status(204).send();
    });
});

// View Employee
app.get('/employees/:id', (req, res) => {
    Employee.findById(req.params.id, (err, employee) => {
        if (err) res.status(500).send(err);
        res.json(employee);
    });
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
